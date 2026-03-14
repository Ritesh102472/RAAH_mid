"""
Citizen Upload Router
Handles image/video upload → EXIF extraction → AI detection → complaint filing
"""
import re
import math
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import User, Pothole, Complaint, Report, Severity, PotholeStatus, ComplaintStatus, ReportSource
from auth.dependencies import get_current_user, get_optional_user
from utils.exif import extract_gps_from_image
from utils.file_storage import save_file, get_file_url
from ai_service.model import run_detection, classify_severity_from_detection
from map_service.geocoding import reverse_geocode
import asyncio
import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/citizen", tags=["citizen"])

MERGE_RADIUS_METERS = 15.0


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def assign_agency(state: str, road_name: Optional[str]) -> str:
    """Assign maintenance agency based on road type."""
    road_upper = (road_name or "Unknown").upper()
    if any(tag in road_upper for tag in ["NH-", "NATIONAL HIGHWAY"]):
        return "NHAI"
    elif any(tag in road_upper for tag in ["SH-", "STATE HIGHWAY"]):
        return f"PWD {state or 'State'}"
    return "Municipal Corporation"


def generate_complaint_number(pothole_id: int) -> str:
    from datetime import date
    return f"CMP-{date.today().year}-{pothole_id:04d}"


def find_nearby_pothole(db: Session, lat: float, lng: float, exclude_ids: list[int] = None) -> Optional[Pothole]:
    """Find an existing pothole within MERGE_RADIUS_METERS, excluding specified IDs."""
    if exclude_ids is None:
        exclude_ids = []
    potholes = db.query(Pothole).filter(
        Pothole.status != PotholeStatus.resolved
    ).all()
    for p in potholes:
        if p.id in exclude_ids:
            continue
        dist = haversine_distance(lat, lng, p.lat, p.lng)
        if dist <= MERGE_RADIUS_METERS:
            return p
    return None


@router.post("/upload")
async def upload_pothole(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    try:
        """
        Main upload endpoint:
        1. Save file
        2. Extract GPS from EXIF (or use provided lat/lng)
        3. Run AI detection
        4. Reverse geocode
        5. Create/merge pothole record
        6. Auto-generate complaint
        """
        ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/jpg", "video/mp4", "video/quicktime"}
        if file.content_type not in ALLOWED:
            raise HTTPException(400, detail=f"Unsupported file type: {file.content_type}")

        file_bytes = await file.read()
        if len(file_bytes) > 50 * 1024 * 1024:
            raise HTTPException(400, detail="File too large (max 50MB)")

        # 1. Save file
        file_path = await save_file(file_bytes, file.filename)
        file_url = get_file_url(file_path)

        # 2. Extract GPS
        exif_coords = None
        if file.content_type.startswith("image/"):
            exif_coords = extract_gps_from_image(file_bytes)

        # Priority logic: EXIF > Browser GPS
        if exif_coords:
            lat, lng = exif_coords
            location_source = "image_exif"
        elif latitude is not None and longitude is not None:
            lat, lng = latitude, longitude
            location_source = "browser_gps"
        else:
            return {
                "status": "location_required",
                "message": "We couldn't detect the location automatically. Please enable GPS or ensure the image has EXIF data.",
                "file_path": file_url,
                "potholes": [],
            }

        # 3. Run AI detection
        detection_result = await run_detection(file_bytes, file.filename)
        potholes_detected = detection_result.get("potholes", [])

        if not potholes_detected:
            return {
                "status": "no_pothole_detected",
                "message": "No potholes detected in this image.",
                "location": {"lat": lat, "lng": lng, "source": location_source},
                "potholes": [],
            }

        # 4. Reverse geocode
        geo = await reverse_geocode(lat, lng)

        # 5. Create/merge potholes
        created_potholes = []
        handled_pothole_ids = []
        for det in potholes_detected:
            bbox = det.get("bbox", [0, 0, 0, 0])
            confidence = det.get("confidence", 0.0)
            sev_str = classify_severity_from_detection(bbox, confidence)
            severity = Severity(sev_str)

            # Try to merge with nearby existing pothole
            nearby = find_nearby_pothole(db, lat, lng, exclude_ids=handled_pothole_ids)

            # CRITICAL: If we find a nearby pothole but it was JUST created in this same upload, 
            # do NOT merge. They are distinct detections in the same image.
            # (Now handled by exclude_ids)
            if nearby:
                handled_pothole_ids.append(nearby.id)
                # Merge — increment report count on complaint
                report = Report(
                    pothole_id=nearby.id,
                    user_id=current_user.id if current_user else None,
                    source=ReportSource.citizen_upload,
                    image_path=file_path,
                    lat=lat, lng=lng,
                    location_source=location_source,
                )
                db.add(report)
                if nearby.complaint:
                    nearby.complaint.number_of_reports += 1
                    nearby.complaint.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                
                created_potholes.append({
                    "pothole_id": nearby.id,
                    "merged": True,
                    "severity": sev_str,
                    "confidence": confidence,
                    "bbox": bbox,
                    "lat": lat, "lng": lng,
                })
            else:
                # New pothole
                pothole = Pothole(
                    lat=lat, lng=lng,
                    road_name=geo["road"],
                    city=geo["city"],
                    state=geo["state"],
                    country=geo.get("country", "India"),
                    full_address=geo.get("full_address"),
                    location_source=location_source,
                    image_path=file_path,
                    severity=severity,
                    confidence=confidence,
                    bbox_x=bbox[0] if len(bbox) > 0 else None,
                    bbox_y=bbox[1] if len(bbox) > 1 else None,
                    bbox_w=bbox[2] if len(bbox) > 2 else None,
                    bbox_h=bbox[3] if len(bbox) > 3 else None,
                    status=PotholeStatus.detected,
                    source=ReportSource.citizen_upload,
                    reporter_id=current_user.id if current_user else None,
                )
                db.add(pothole)
                db.flush() # Get ID for new_pothole_ids and complaint
                handled_pothole_ids.append(pothole.id)

                # 6. Auto-generate complaint
                agency = assign_agency(geo["state"], geo["road"])
                complaint = Complaint(
                    complaint_number=generate_complaint_number(pothole.id),
                    pothole_id=pothole.id,
                    lat=lat,
                    lng=lng,
                    location_text=f"{geo['road']}, {geo['city']}, {geo['state']}",
                    road_name=geo["road"],
                    severity=severity,
                    number_of_reports=1,
                    status=ComplaintStatus.reported,
                    agency=agency,
                )
                db.add(complaint)
                pothole.status = PotholeStatus.complaint_filed

                # Initial report
                report = Report(
                    pothole_id=pothole.id,
                    user_id=current_user.id if current_user else None,
                    source=ReportSource.citizen_upload,
                    image_path=file_path,
                    lat=lat, lng=lng,
                    location_source=location_source,
                )
                db.add(report)

                created_potholes.append({
                    "pothole_id": pothole.id,
                    "merged": False,
                    "severity": sev_str,
                    "confidence": confidence,
                    "bbox": bbox,
                    "lat": lat, "lng": lng,
                    "complaint_number": complaint.complaint_number,
                })

        db.commit()

        return {
            "status": "success",
            "location": {"lat": lat, "lng": lng, "source": location_source},
            "road": geo["road"],
            "city": geo["city"],
            "potholes": created_potholes,
            "total_detected": len(potholes_detected),
            "file_url": file_url,
        }
    except Exception as e:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"\n\nERROR at {datetime.datetime.now()}:\n")
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/potholes/{pothole_id}")
def get_pothole_detail(pothole_id: int, db: Session = Depends(get_db)):
    p = db.query(Pothole).filter(Pothole.id == pothole_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pothole not found")
    
    return {
        "id": f"PTH-{p.id:04d}",
        "latitude": p.lat,
        "longitude": p.lng,
        "road": p.road_name,
        "city": p.city,
        "state": p.state,
        "country": p.country,
        "address": p.full_address or f"{p.road_name}, {p.city}, {p.state}, {p.country}",
        "severity": p.severity,
        "confidence": p.confidence,
        "location_source": p.location_source
    }

@router.post("/mapillary/scan")
async def trigger_mapillary_scan(
    latitude: float,
    longitude: float,
    radius: int = 50,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Trigger a background Celery task to scan the specified area using Mapillary imagery.
    """
    from utils.tasks import scan_mapillary_area
    user_id = current_user.id if current_user else None
    
    # Dispatch the Celery task
    task = scan_mapillary_area.delay(latitude, longitude, radius, user_id)
    
    return {
        "status": "queued",
        "message": f"Mapillary scan queued for {latitude}, {longitude} with radius {radius}m.",
        "task_id": str(task.id)
    }
