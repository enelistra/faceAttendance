from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json
import cv2
import numpy as np
import base64
import time
from .models import Employee, Attendance, ExternalEmployeeMap
from .face_utils import face_system


# -----------------------------
# Helpers
# -----------------------------

def _make_json_response(payload, status=200, allow_iframe=True):
    """Return JsonResponse with headers that allow iframe embedding when needed."""
    resp = JsonResponse(payload, status=status)
    if allow_iframe:
        resp["X-Frame-Options"] = "ALLOWALL"
        resp["Content-Security-Policy"] = "frame-ancestors *"
    # Add CORS headers for API endpoints
    resp["Access-Control-Allow-Origin"] = "*"
    resp["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    resp["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return resp


def _make_render_response(request, template, context=None):
    """Render template and add headers to allow iframe embedding (useful for local dev)."""
    context = context or {}
    resp = render(request, template, context)
    resp["X-Frame-Options"] = "ALLOWALL"
    resp["Content-Security-Policy"] = "frame-ancestors *"
    # Add CORS headers for API endpoints
    resp["Access-Control-Allow-Origin"] = "*"
    resp["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    resp["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return resp


def _parse_json_request(request):
    """Safely parse JSON request body and return dict (or empty dict on failure).
    This handles content-type variations (charset) and empty bodies.
    """
    try:
        content_type = request.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/json"):
            return None, 'Not JSON'

        body = request.body
        if not body:
            return None, 'Empty body'

        # decode bytes -> str (json.loads accepts bytes in py3.11 but be explicit)
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")

        data = json.loads(body)
        return data, None
    except json.JSONDecodeError as e:
        return None, f'JSON decode error: {str(e)}'
    except Exception as e:
        return None, f'Unknown error parsing JSON: {str(e)}'


def _decode_base64_image(image_data):
    """Accept either a raw base64 string or a data URL (data:image/..;base64,...)
    Return OpenCV image (BGR) or raise ValueError on failure.
    """
    if not image_data:
        raise ValueError("No image data provided")

    # If data URL, remove prefix
    if isinstance(image_data, str) and "," in image_data:
        image_data = image_data.split(",", 1)[1]

    # Fix padding if missing
    image_data = image_data.strip()
    missing_padding = len(image_data) % 4
    if missing_padding:
        image_data += '=' * (4 - missing_padding)

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise ValueError(f"Base64 decode error: {str(e)}")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from bytes")
    return image


# -----------------------------
# Views (with Accurate ArcFace Integration)
# -----------------------------

@csrf_exempt
def api_map_employee(request):
    """
    API for external systems to map employee data and get registration URL
    """
    print("🎯 === API Map Employee Request ===")

    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        print("✅ Handling OPTIONS preflight request")
        return _make_json_response({
            "success": True,
            "message": "CORS preflight OK"
        })

    if request.method != 'POST':
        print("❌ Method not allowed")
        return _make_json_response({
            "success": False,
            "message": "POST method required"
        }, status=405)

    data, err = _parse_json_request(request)
    if err:
        print(f"❌ {err}")
        return _make_json_response({"success": False, "message": err}, status=400)

    # FIX: Convert to string before calling strip() to handle both string and integer inputs
    employee_id = str(data.get('employee_id') or '').strip()
    employee_name = str(data.get('employee_name') or '').strip()

    print(f"📥 Received data - Employee ID: {employee_id}, Name: {employee_name}")

    if not employee_id or not employee_name:
        print("❌ Missing employee ID or name")
        return _make_json_response({
            "success": False,
            "message": "Employee ID and name are required"
        }, status=400)

    try:
        # Check if mapping already exists (by employee_id OR employee_name)
        existing_mapping = ExternalEmployeeMap.objects.filter(
            employee_id=employee_id,
            employee_name=employee_name
        ).first()

        if existing_mapping:
            if existing_mapping.is_registered:
                print(f"⚠️ Employee already registered: {employee_id}")
                return _make_json_response({
                    "success": False,
                    "message": "Employee is already registered"
                }, status=409)
            else:
                entry = existing_mapping
                print(f"🔄 Using existing mapping: {entry.map_id}")
        else:
            entry = ExternalEmployeeMap.objects.create(
                employee_id=employee_id,
                employee_name=employee_name
            )
            print(f"✅ Created new mapping: {entry.map_id}")

        redirect_url = f"/face-register/?id={entry.map_id}"
        full_redirect_url = request.build_absolute_uri(redirect_url)

        response_data = {
            "success": True,
            "map_id": entry.map_id,
            "redirect_url": full_redirect_url,
            "registration_url": redirect_url,
            "message": "Employee mapped successfully"
        }

        print(f"📤 Sending response: {response_data}")
        return _make_json_response(response_data)

    except Exception as e:
        print(f"❌ API map employee error: {e}")
        return _make_json_response({
            "success": False,
            "message": f"Server error: {str(e)}"
        }, status=500)


@csrf_exempt
def api_mark_attendance(request):
    """
    API for external systems to mark attendance with face verification
    """
    print("🎯 === API Mark Attendance Request ===")

    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        print("✅ Handling OPTIONS preflight request for attendance")
        return _make_json_response({
            "success": True,
            "message": "CORS preflight OK"
        })


    if request.method != 'POST':
        print("❌ Method not allowed")
        return _make_json_response({"success": False, "message": "POST method required"}, status=405)
    

    data, err = _parse_json_request(request)
    if err:
        print(f"❌ {err}")
        return _make_json_response({"success": False, "message": err}, status=400)

    # FIX: Convert to string before calling strip()
    employee_id = str(data.get('employee_id') or '').strip()
    image_data = data.get('image_data')
    timestamp = data.get('timestamp', None)

    print(f"📥 Received attendance request - Employee ID: {employee_id}")
    

    if not employee_id or not image_data:
        print("❌ Missing employee ID or image data")
        return _make_json_response({
            "success": False,
            "message": "Employee ID and image data are required"
        }, status=400)

    # Find employee
    try:
        employee = Employee.objects.get(employee_id=employee_id, is_active=True)
        print(f"✅ Employee found: {employee.employee_name}")
    except Employee.DoesNotExist:
        print(f"❌ Employee not found or not registered: {employee_id}")
        return _make_json_response({
            "success": False,
            "message": "Employee not found or not registered. Please register first."
        }, status=404)

    # Decode image
    try:
        image = _decode_base64_image(image_data)
    except ValueError as e:
        print(f"❌ Image processing error: {e}")
        return _make_json_response({"success": False, "message": str(e)}, status=400)

    print(f"🖼️ Processing attendance image: {image.shape}")

    # Face detection with accurate ArcFace
    detection_result = face_system.detect_face_in_frame(image)

    if not detection_result.get('face_detected'):
        print("❌ No face detected")
        return _make_json_response({"success": False, "message": "No face detected in the image"}, status=400)

    if detection_result.get('spoofing_detected', False):
        print(f"❌ Spoofing detected: {detection_result.get('message')}")
        return _make_json_response({
            "success": False, 
            "message": f"🚫 SPOOFING DETECTED: {detection_result.get('message')}",
            "consecutive_spoofs": detection_result.get('consecutive_spoofs', 0)
        }, status=403)

    if detection_result.get('quality_issue', False):
        print(f"❌ Face quality issue: {detection_result.get('message')}")
        return _make_json_response({"success": False, "message": f"Face quality issue: {detection_result.get('message')}"}, status=400)

    # Additional Gemini spoofing check as backup
    print("🤖 Starting Gemini spoofing check...")
    try:
        is_genuine, spoofing_message = face_system.check_spoofing_with_gemini(image)
    except Exception as e:
        print(f"❌ Spoofing check failed: {e}")
        return _make_json_response({"success": False, "message": f"Spoofing check failed: {str(e)}"}, status=500)

    if not is_genuine:
        print(f"❌ Spoofing detected: {spoofing_message}")
        return _make_json_response({"success": False, "message": f"SPOOFING DETECTED: {spoofing_message}"}, status=403)

    # Face comparison with registered employee using accurate ArcFace
    stored_patterns = employee.get_face_encoding()
    if not stored_patterns:
        print(f"❌ No face patterns stored for employee: {employee_id}")
        return _make_json_response({"success": False, "message": "No face patterns found for this employee. Please re-register."}, status=400)

    is_match, similarity = face_system.compare_faces(stored_patterns, detection_result.get('facial_patterns'))

    print(f"🔍 Face comparison result - Match: {is_match}, Similarity: {similarity:.3f}")

    if not is_match:
        print(f"❌ Face doesn't match registered employee")
        return _make_json_response({
            "success": False, 
            "message": f"Face doesn't match registered employee. Similarity: {similarity*100:.1f}% (required: {face_system.face_match_threshold*100:.1f}%)"
        }, status=401)

    # Check if already marked today
    today = timezone.now().date()
    existing_attendance = Attendance.objects.filter(employee=employee, timestamp__date=today).first()

    if existing_attendance:
        print(f"ℹ️ Attendance already marked today at {existing_attendance.timestamp}")
        return _make_json_response({
            "success": True,
            "message": f"Attendance already marked today at {existing_attendance.timestamp.strftime('%H:%M:%S')}",
            "employee_id": employee.employee_id,
            "employee_name": employee.employee_name,
            "timestamp": existing_attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "similarity": round(similarity, 3),
            "confidence": detection_result.get('confidence'),
            "security_score": detection_result.get('security_score'),
            "already_marked": True
        })

    # Record attendance
    attendance = Attendance.objects.create(employee=employee)
    print(f"✅ Attendance marked successfully for {employee.employee_name}")

    response_data = {
        "success": True,
        "message": f"✅ Attendance marked successfully for {employee.employee_name}!",
        "employee_id": employee.employee_id,
        "employee_name": employee.employee_name,
        "timestamp": attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "similarity": round(similarity, 3),
        "confidence": detection_result.get('confidence'),
        "security_score": detection_result.get('security_score'),
        "already_marked": False
    }

    print(f"📤 Sending response: {response_data}")
    return _make_json_response(response_data)


@csrf_exempt
def face_register(request):
    """
    Real-time face registration with continuous scanning
    """
    print("🎯 === Face Registration Page Accessed ===")

    map_id = request.GET.get("id")
    mapped_data = None

    print(f"🔍 Checking for map_id in URL: {map_id}")

    if map_id:
        try:
            mapped_data = ExternalEmployeeMap.objects.get(map_id=map_id)
            print(f"📍 Loaded mapped data: {mapped_data.employee_name} ({mapped_data.employee_id}) - Registered: {mapped_data.is_registered}")

            if mapped_data.is_registered:
                print(f"⚠️ Employee already registered: {mapped_data.employee_id}")
        except ExternalEmployeeMap.DoesNotExist:
            print(f"❌ No mapping found for map_id: {map_id}")
            mapped_data = None
        except Exception as e:
            print(f"❌ Error loading mapped data: {e}")
            mapped_data = None
    else:
        print("📍 Manual registration mode - no map_id provided")

    if request.method == 'POST':
        print("📨 POST request received for face registration API")

        data, err = _parse_json_request(request)
        if err:
            # If not JSON, render the page (legacy behavior)
            print("⚠️ POST received but not JSON - rendering template")
            return _make_render_response(request, 'face_register.html', {"mapped_data": mapped_data})

        request_type = data.get('type', 'scan_frame')

        if request_type == 'scan_frame':
            return handle_frame_scan(data)
        elif request_type == 'verify_face':
            return handle_face_verification(data)
        elif request_type == 'final_registration':
            return handle_final_registration(data)
        else:
            return _make_json_response({'success': False, 'message': 'Invalid request type'}, status=400)

    # GET
    print(f"🎨 Rendering template with mapped_data: {mapped_data is not None}")
    return _make_render_response(request, 'face_register.html', {"mapped_data": mapped_data})


def handle_frame_scan(data):
    try:
        image_data = data.get('image_data', '')
        if not image_data:
            return _make_json_response({'success': False, 'face_detected': False, 'message': 'No image data provided'}, status=400)

        image = _decode_base64_image(image_data)
        print(f"🖼️ Processing frame: {image.shape}")
        detection_result = face_system.detect_face_in_frame(image)
        return _make_json_response(detection_result)
    except Exception as e:
        print(f"❌ Frame scan error: {e}")
        return _make_json_response({'success': False, 'face_detected': False, 'message': f'Scanning error: {str(e)}'}, status=500)


def handle_face_verification(data):
    try:
        image_data = data.get('image_data', '')
        if not image_data:
            return _make_json_response({'success': False, 'verified': False, 'message': 'No image data provided for verification'}, status=400)

        image = _decode_base64_image(image_data)

        print("🤖 Starting FOOLPROOF face verification...")
        detection_result = face_system.detect_face_in_frame(image)

        if not detection_result.get('face_detected'):
            return _make_json_response({'success': False, 'verified': False, 'message': 'No face detected for verification'}, status=400)

        if detection_result.get('spoofing_detected', False):
            return _make_json_response({
                'success': False, 
                'verified': False, 
                'message': f'🚫 SPOOFING DETECTED: {detection_result.get("message")}',
                'consecutive_spoofs': detection_result.get('consecutive_spoofs', 0)
            }, status=403)

        if detection_result.get('quality_issue', False):
            return _make_json_response({'success': False, 'verified': False, 'message': f'Face quality issue: {detection_result.get("message")}'}, status=400)

        # Additional Gemini check as backup
        is_genuine, spoofing_message = face_system.check_spoofing_with_gemini(image)

        if is_genuine:
            return _make_json_response({
                'success': True, 
                'verified': True, 
                'message': '✅ Face verified successfully with FOOLPROOF security! You can now register.', 
                'facial_patterns': detection_result.get('facial_patterns'), 
                'face_location': detection_result.get('face_location'),
                'confidence': detection_result.get('confidence'),
                'security_score': detection_result.get('security_score')
            })
        else:
            return _make_json_response({'success': False, 'verified': False, 'message': f'❌ Verification failed: {spoofing_message}'}, status=403)

    except Exception as e:
        print(f"❌ Face verification error: {e}")
        return _make_json_response({'success': False, 'verified': False, 'message': f'Verification error: {str(e)}'}, status=500)


def handle_final_registration(data):
    try:
        map_id = data.get('map_id')
        facial_patterns = data.get('facial_patterns')

        print(f"🔍 Processing final registration with map_id: {map_id}")

        if map_id:
            try:
                mapping = ExternalEmployeeMap.objects.get(map_id=map_id)
                if mapping.is_registered:
                    return _make_json_response({'success': False, 'message': 'This employee is already registered'}, status=409)
                employee_id = mapping.employee_id
                employee_name = mapping.employee_name
                print(f"👤 Mapped registration for: {employee_name} ({employee_id})")
            except ExternalEmployeeMap.DoesNotExist:
                return _make_json_response({'success': False, 'message': 'Invalid mapping ID'}, status=400)

        else:
            # FIX: Convert to string before calling strip()
            employee_id = str(data.get('employee_id') or '').strip()
            employee_name = str(data.get('employee_name') or '').strip()
            if not employee_id or not employee_name:
                return _make_json_response({'success': False, 'message': 'Employee ID and name are required'}, status=400)

        if not facial_patterns:
            return _make_json_response({'success': False, 'message': 'Facial patterns are required for registration'}, status=400)

        print(f"👤 Final registration for: {employee_name} ({employee_id})")

        if Employee.objects.filter(employee_id=employee_id).exists():
            return _make_json_response({'success': False, 'message': f'Employee ID {employee_id} already exists'}, status=409)

        try:
            employee = Employee(employee_id=employee_id, employee_name=employee_name)
            employee.set_face_encoding(facial_patterns)
            employee.save()

            if map_id:
                mapping.is_registered = True
                mapping.save()
                print(f"✅ Mapping updated for map_id: {map_id}")

            print(f"✅ Employee {employee_name} saved successfully!")
            return _make_json_response({
                'success': True, 
                'message': f'✅ Employee {employee_name} registered successfully with FOOLPROOF security!', 
                'employee_id': employee_id, 
                'employee_name': employee_name, 
                'map_id': map_id
            })

        except Exception as e:
            print(f"❌ Database error: {e}")
            return _make_json_response({'success': False, 'message': f'Database error: {str(e)}'}, status=500)

    except Exception as e:
        print(f"❌ Final registration error: {e}")
        return _make_json_response({'success': False, 'message': f'Registration error: {str(e)}'}, status=500)


@csrf_exempt
def mark_attendance(request):
    """
    Real-time attendance marking with continuous face scanning
    """
    print("🎯 === Mark Attendance Request ===")
    print(f"📝 Request method: {request.method}")
    print(f"📝 Content type: {request.content_type}")

    if request.method == 'GET':
        print("📄 Rendering mark attendance page (GET request)")
        return _make_render_response(request, 'mark_attendance.html')

    elif request.method == 'POST':
        print("📨 POST request received for attendance")
        data, err = _parse_json_request(request)
        if err:
            print("⚠️ POST received but not JSON - rendering page")
            return _make_render_response(request, 'mark_attendance.html')

        request_type = data.get('type', 'scan_frame')
        print(f"🔍 Processing request type: {request_type}")

        if request_type == 'scan_frame':
            return handle_attendance_scan(data)
        elif request_type == 'verify_attendance':
            return handle_attendance_verification(data)
        else:
            return _make_json_response({'success': False, 'message': 'Invalid request type'}, status=400)

    else:
        print(f"❌ Method not allowed: {request.method}")
        return _make_json_response({'success': False, 'message': 'GET or POST method required'}, status=405)


def handle_attendance_scan(data):
    try:
        image_data = data.get('image_data', '')
        if not image_data:
            return _make_json_response({'success': False, 'face_detected': False, 'message': 'No image data provided'}, status=400)

        image = _decode_base64_image(image_data)
        print(f"🖼️ Processing attendance frame: {image.shape}")
        detection_result = face_system.detect_face_in_frame(image)
        return _make_json_response(detection_result)

    except Exception as e:
        print(f"❌ Attendance scan error: {e}")
        return _make_json_response({'success': False, 'face_detected': False, 'message': f'Scanning error: {str(e)}'}, status=500)


def handle_attendance_verification(data):
    try:
        image_data = data.get('image_data', '')
        if not image_data:
            return _make_json_response({ 'success': False, 'message': 'No image data provided' }, status=400)

        image = _decode_base64_image(image_data)
        print("🔍 Final attendance verification with FOOLPROOF system...")

        detection_result = face_system.detect_face_in_frame(image)

        if not detection_result.get('face_detected'):
            return _make_json_response({'success': False, 'message': 'Face not detected for attendance'}, status=400)

        if detection_result.get('spoofing_detected', False):
            return _make_json_response({
                'success': False, 
                'message': f'🚫 SPOOFING DETECTED: {detection_result.get("message")}',
                'consecutive_spoofs': detection_result.get('consecutive_spoofs', 0)
            }, status=403)

        if detection_result.get('quality_issue', False):
            return _make_json_response({'success': False, 'message': f'Face quality issue: {detection_result.get("message")}'}, status=400)

        print("🤖 Running additional Gemini spoofing check...")
        is_genuine, spoofing_message = face_system.check_spoofing_with_gemini(image)

        if not is_genuine:
            return _make_json_response({'success': False, 'message': f'SPOOFING DETECTED: {spoofing_message}'}, status=403)

        employees = Employee.objects.filter(is_active=True)
        
        # Prepare data for ArcFace comparison
        stored_encodings = []
        stored_names = []
        stored_ids = []
        
        for employee in employees:
            stored_patterns = employee.get_face_encoding()
            if stored_patterns:
                stored_encodings.append(stored_patterns)
                stored_names.append(employee.employee_name)
                stored_ids.append(employee.employee_id)

        print(f"🔍 Comparing with {len(stored_encodings)} registered employees using ArcFace...")

        # Use the accurate ArcFace comparison
        current_embedding = detection_result.get('facial_patterns')
        matched_name, similarity, matched_id = face_system.compare_faces_arcface(
            current_embedding, stored_encodings, stored_names, stored_ids
        )

        if matched_name and matched_id:
            matched_employee = Employee.objects.get(employee_id=matched_id)
            today = timezone.now().date()
            existing_attendance = Attendance.objects.filter(employee=matched_employee, timestamp__date=today).first()

            if existing_attendance:
                print(f"ℹ️ Attendance already marked today at {existing_attendance.timestamp}")
                return _make_json_response({
                    'success': True,
                    'message': f'✅ Attendance already marked today for {matched_employee.employee_name} at {existing_attendance.timestamp.strftime("%H:%M:%S")}!',
                    'employee_id': matched_employee.employee_id,
                    'employee_name': matched_employee.employee_name,
                    'timestamp': existing_attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'similarity': round(similarity, 3),
                    'confidence': detection_result.get('confidence'),
                    'security_score': detection_result.get('security_score'),
                    'already_marked': True
                })

            attendance = Attendance.objects.create(employee=matched_employee)
            print(f"✅ Attendance marked for {matched_employee.employee_name}")

            return _make_json_response({
                'success': True,
                'message': f'✅ Attendance marked for {matched_employee.employee_name}!\n\nFace Match: ✅ ({similarity*100:.1f}%)\nSpoofing Check: ✅\nSecurity Score: {detection_result.get("security_score", 0):.2f}\nTime: {timezone.now().strftime("%H:%M:%S")}',
                'employee_id': matched_employee.employee_id,
                'employee_name': matched_employee.employee_name,
                'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'similarity': round(similarity, 3),
                'confidence': detection_result.get('confidence'),
                'security_score': detection_result.get('security_score'),
                'already_marked': False
            })

        else:
            return _make_json_response({
                'success': False, 
                'message': f'No matching employee found. Similarity: {similarity*100:.1f}% (required: {face_system.face_match_threshold*100:.1f}%)'
            }, status=404)

    except Exception as e:
        print(f"❌ Attendance verification error: {e}")
        return _make_json_response({'success': False, 'message': f'Verification error: {str(e)}'}, status=500)


@csrf_exempt
def view_database(request):
    """
    View registered employees database
    """
    print("🎯 === View Database Request ===")
    
    employees = Employee.objects.filter(is_active=True)
    employee_list = []
    
    for employee in employees:
        employee_list.append({
            'employee_id': employee.employee_id,
            'employee_name': employee.employee_name,
            'registration_date': employee.registration_date.strftime('%Y-%m-%d %H:%M:%S') if employee.registration_date else 'N/A'
        })
    
    return _make_json_response({
        'success': True,
        'employees': employee_list,
        'total_count': len(employee_list)
    })


@csrf_exempt
def view_attendance_log(request):
    """
    View attendance log
    """
    print("🎯 === View Attendance Log Request ===")
    
    date_filter = request.GET.get('date')
    employee_id_filter = request.GET.get('employee_id')
    
    attendance_records = Attendance.objects.all().order_by('-timestamp')
    
    if date_filter:
        try:
            target_date = timezone.datetime.strptime(date_filter, '%Y-%m-%d').date()
            attendance_records = attendance_records.filter(timestamp__date=target_date)
        except ValueError:
            return _make_json_response({'success': False, 'message': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)
    
    if employee_id_filter:
        attendance_records = attendance_records.filter(employee__employee_id=employee_id_filter)
    
    attendance_list = []
    
    for record in attendance_records:
        attendance_list.append({
            'employee_id': record.employee.employee_id,
            'employee_name': record.employee.employee_name,
            'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return _make_json_response({
        'success': True,
        'attendance_records': attendance_list,
        'total_count': len(attendance_list)
    })


@csrf_exempt
def system_status(request):
    """
    Get system status and statistics
    """
    print("🎯 === System Status Request ===")
    
    total_employees = Employee.objects.filter(is_active=True).count()
    total_attendance_today = Attendance.objects.filter(timestamp__date=timezone.now().date()).count()
    total_mappings = ExternalEmployeeMap.objects.count()
    registered_mappings = ExternalEmployeeMap.objects.filter(is_registered=True).count()
    
    return _make_json_response({
        'success': True,
        'system_status': 'OPERATIONAL',
        'face_system_ready': face_system.face_app is not None,
        'gemini_ready': face_system.model is not None,
        'statistics': {
            'total_employees': total_employees,
            'attendance_today': total_attendance_today,
            'total_mappings': total_mappings,
            'registered_mappings': registered_mappings
        },
        'security_features': {
            'foolproof_anti_spoofing': True,
            'gemini_ai_verification': face_system.model is not None,
            'arcface_accuracy': True,
            'multi_layer_security': True
        }
    })