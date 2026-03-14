"""
Celery Background Tasks
"""
import datetime
from celery_app import celery_app


@celery_app.task(name="utils.tasks.auto_escalate_complaints")
def auto_escalate_complaints():
    """
    Auto-escalate complaints that have been pending for more than 7 days.
    Runs daily via Celery Beat.
    """
    from database.connection import SessionLocal
    from database.models import Complaint, Pothole, ComplaintStatus, PotholeStatus
    db = SessionLocal()
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        pending = db.query(Complaint).filter(
            Complaint.status.in_([ComplaintStatus.filed, ComplaintStatus.pending]),
            Complaint.created_at < cutoff,
        ).all()

        count = 0
        for c in pending:
            c.status = ComplaintStatus.escalated
            c.escalated_at = datetime.datetime.utcnow()
            c.updated_at = datetime.datetime.utcnow()
            if c.pothole:
                c.pothole.status = PotholeStatus.escalated
            count += 1
        db.commit()
        print(f"[Celery] Auto-escalated {count} complaints")
        return {"escalated": count}
    except Exception as e:
        db.rollback()
        print(f"[Celery] Escalation failed: {e}")
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="utils.tasks.refresh_predictions")
def refresh_predictions():
    """Refresh PVI predictions every 6 hours."""
    import asyncio
    from database.connection import SessionLocal
    from database.models import Prediction
    from prediction_service.pvi import compute_predictions_for_grid

    db = SessionLocal()
    try:
        predictions = asyncio.run(compute_predictions_for_grid())
        db.query(Prediction).delete()
        for pred in predictions:
            p = Prediction(
                lat=pred["lat"],
                lng=pred["lng"],
                pvi_score=pred["pvi_score"],
                risk_level=pred["risk_level"],
                road_type=pred["road_type"],
                rainfall_mm=pred["rainfall_mm"],
                temperature_c=pred["temperature_c"],
                computed_at=datetime.datetime.utcnow(),
            )
            db.add(p)
        db.commit()
        print(f"[Celery] Refreshed {len(predictions)} predictions")
        return {"count": len(predictions)}
    except Exception as e:
        db.rollback()
        print(f"[Celery] Prediction refresh failed: {e}")
        return {"error": str(e)}
    finally:
        db.close()

@celery_app.task(name="utils.tasks.scan_mapillary_area")
def scan_mapillary_area(lat: float, lng: float, radius: int = 50, user_id: int = None):
    """
    Fetch Mapillary images in the area, run inference on them, and record potholes.
    """
    import asyncio
    from database.connection import SessionLocal
    from database.models import Pothole, Report, Severity, PotholeStatus, ReportSource
    from map_service.mapillary import fetch_nearby_images, get_image_by_id
    from map_service.geocoding import reverse_geocode
    from ai_service.model import run_detection, classify_severity_from_detection
    from citizen_service.router import find_nearby_pothole
    import datetime

    db = SessionLocal()
    try:
        # Run async fetches in the synchronous celery worker
        images = asyncio.run(fetch_nearby_images(lat, lng, radius))
        if not images:
            print(f"[Mapillary] No images found near {lat}, {lng}")
            return {"status": "no_images", "count": 0}

        print(f"[Mapillary] Found {len(images)} images to process")
        total_potholes = 0
        handled_pothole_ids = []

        for p_img in images:
            img_id = p_img.get("id")
            if not img_id:
                continue
                
            geom = p_img.get("geometry", {})
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                # Default to requested coordinate if imagery lacks exact point
                img_lng, img_lat = lng, lat
            else:
                img_lng, img_lat = coords[0], coords[1]

            # Fetch actual bytes
            img_bytes = asyncio.run(get_image_by_id(img_id))
            if not img_bytes:
                continue

            # Run detection
            detection_result = asyncio.run(run_detection(img_bytes, f"mapillary_{img_id}.jpg"))
            potholes_detected = detection_result.get("potholes", [])
            
            if not potholes_detected:
                continue

            # Reverse geocode for location text
            geo = asyncio.run(reverse_geocode(img_lat, img_lng))

            for det in potholes_detected:
                bbox = det.get("bbox", [0, 0, 0, 0])
                confidence = det.get("confidence", 0.0)
                sev_str = classify_severity_from_detection(bbox, confidence)
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = Severity.medium

                # Check if this pothole was already reported / detected nearby
                nearby = find_nearby_pothole(db, img_lat, img_lng, exclude_ids=handled_pothole_ids)

                if nearby:
                    handled_pothole_ids.append(nearby.id)
                    # Add report
                    try:
                        report = Report(
                            pothole_id=nearby.id,
                            user_id=user_id,
                            source=ReportSource.mapillary,
                            image_path=f"mapillary://{img_id}",  # Storing mapillary ID as path
                            lat=img_lat, lng=img_lng,
                            location_source="mapillary_api",
                        )
                        db.add(report)
                        if nearby.complaint:
                            nearby.complaint.number_of_reports += 1
                            nearby.complaint.updated_at = datetime.datetime.utcnow()
                    except Exception as report_e:
                        print(f"Failed to add report: {report_e}")
                else:
                    # New pothole
                    try:
                        pothole = Pothole(
                            lat=img_lat, lng=img_lng,
                            road_name=geo.get("road", "Unknown Road"),
                            city=geo.get("city", "Unknown City"),
                            state=geo.get("state"),
                            country=geo.get("country", "India"),
                            full_address=geo.get("full_address"),
                            location_source="mapillary_api",
                            image_path=f"mapillary://{img_id}",
                            severity=severity,
                            confidence=confidence,
                            bbox_x=bbox[0] if len(bbox) > 0 else None,
                            bbox_y=bbox[1] if len(bbox) > 1 else None,
                            bbox_w=bbox[2] if len(bbox) > 2 else None,
                            bbox_h=bbox[3] if len(bbox) > 3 else None,
                            status=PotholeStatus.detected,
                            source=ReportSource.mapillary,
                            reporter_id=user_id,
                        )
                        db.add(pothole)
                        db.commit() # Commit right away to get the ID
                        handled_pothole_ids.append(pothole.id)
                    except Exception as pothole_e:
                        db.rollback()
                        print(f"Failed to add new pothole: {pothole_e}")

                total_potholes += 1

        db.commit()
        print(f"[Mapillary] Scan complete. Found {total_potholes} potholes.")
        return {"status": "success", "potholes_found": total_potholes}

    except Exception as e:
        db.rollback()
        import traceback
        print(f"[Mapillary] Scan task failed: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}
    finally:
        db.close()

