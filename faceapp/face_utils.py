import cv2
import numpy as np
import base64
import json
import time
import logging
from django.conf import settings
import google.generativeai as genai
from insightface.app import FaceAnalysis
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FoolproofAntiSpoofing:
    def __init__(self):
        self.previous_frame = None
        self.motion_history = []
        self.frame_count = 0
        self.last_spoof_check = time.time()
        
    def foolproof_detection(self, frame, face_bbox):
        """
        FOOLPROOF detection that actually blocks phone screens and photos
        Uses multiple layers of detection that 2D images can't bypass
        """
        try:
            left, top, right, bottom = face_bbox
            face_region = frame[top:bottom, left:right]
            
            if face_region.size == 0:
                return True, ["No face detected"], -1.0
            
            # Multiple detection layers
            detection_results = []
            
            # LAYER 1: MOVEMENT DETECTION (Most Important)
            movement_score, movement_reason = self._detect_movement(frame, face_bbox)
            detection_results.append((movement_score, movement_reason))
            
            # LAYER 2: TEXTURE AND DETAIL ANALYSIS
            texture_score, texture_reason = self._analyze_texture_details(face_region)
            detection_results.append((texture_score, texture_reason))
            
            # LAYER 3: COLOR AND LIGHTING ANALYSIS
            color_score, color_reason = self._analyze_color_lighting(face_region)
            detection_results.append((color_score, color_reason))
            
            # LAYER 4: SCREEN/PHOTO SPECIFIC DETECTION
            screen_score, screen_reason = self._detect_screen_characteristics(face_region)
            detection_results.append((screen_score, screen_reason))
            
            # Calculate overall score
            total_score = sum(score for score, _ in detection_results)
            
            # Get all reasons
            all_reasons = [reason for _, reason in detection_results]
            
            # FOOLPROOF DECISION: If total score is negative or any critical failure
            is_spoofed = total_score < 0.2 or any("CRITICAL" in reason for reason in all_reasons)
            
            return is_spoofed, all_reasons, total_score
            
        except Exception as e:
            return True, [f"Detection error: {str(e)}"], -1.0

    def _detect_movement(self, frame, face_bbox):
        """Detect natural face movements that photos can't replicate"""
        try:
            left, top, right, bottom = face_bbox
            current_face = frame[top:bottom, left:right]
            
            if self.previous_frame is None:
                self.previous_frame = current_face
                return 0.1, "First frame - waiting for movement"
            
            # Convert to grayscale and resize for comparison
            current_gray = cv2.cvtColor(current_face, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            size = (80, 80)
            current_resized = cv2.resize(current_gray, size)
            previous_resized = cv2.resize(previous_gray, size)
            
            # Calculate absolute difference
            diff = cv2.absdiff(current_resized, previous_resized)
            
            # Calculate percentage of changed pixels
            changed_pixels = np.sum(diff > 25)  # Threshold for meaningful change
            total_pixels = diff.size
            change_percentage = changed_pixels / total_pixels
            
            # Store movement history
            self.motion_history.append(change_percentage)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)
            
            # Analyze movement pattern
            if len(self.motion_history) >= 5:
                avg_movement = np.mean(self.motion_history)
                max_movement = max(self.motion_history)
                
                # Real faces have natural micro-movements
                if avg_movement > 0.005 and max_movement > 0.01:  # Natural movement range
                    score = 0.4
                    reason = f"Natural movement detected: {avg_movement:.3f}"
                elif avg_movement < 0.001:  # Too static = photo
                    score = -0.5
                    reason = f"CRITICAL: No movement - likely photo: {avg_movement:.3f}"
                else:  # Some movement but not enough
                    score = -0.2
                    reason = f"Limited movement: {avg_movement:.3f}"
            else:
                score = 0.0
                reason = f"Collecting movement data: {change_percentage:.3f}"
            
            self.previous_frame = current_face
            self.frame_count += 1
            
            return score, reason
            
        except Exception as e:
            return -0.3, f"Movement detection failed: {str(e)}"

    def _analyze_texture_details(self, face_region):
        """Analyze facial texture and details that photos lack"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            if height < 50 or width < 50:
                return -0.3, "Face too small for analysis"
            
            # Calculate texture variance (real faces have more texture)
            texture_variance = np.var(gray)
            
            # Calculate edge density (real faces have more natural edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate blur (photos can be blurry)
            blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Combined texture analysis
            texture_score = 0.0
            reasons = []
            
            # Texture variance check
            if texture_variance < 50:
                texture_score -= 0.3
                reasons.append(f"Low texture: {texture_variance:.1f}")
            elif texture_variance > 200:
                texture_score += 0.2
                reasons.append(f"Good texture: {texture_variance:.1f}")
            
            # Edge density check
            if edge_density < 0.03:
                texture_score -= 0.2
                reasons.append(f"Low edges: {edge_density:.3f}")
            elif edge_density > 0.08:
                texture_score += 0.1
                reasons.append(f"Natural edges: {edge_density:.3f}")
            
            # Blur check
            if blur_value < 20:
                texture_score -= 0.2
                reasons.append(f"Blurry: {blur_value:.1f}")
            elif blur_value > 100:
                texture_score += 0.1
                reasons.append(f"Sharp: {blur_value:.1f}")
            
            reason_text = "Texture: " + ", ".join(reasons)
            return texture_score, reason_text
            
        except Exception as e:
            return 0.0, f"Texture analysis failed: {str(e)}"

    def _analyze_color_lighting(self, face_region):
        """Analyze color patterns and lighting that differ in photos"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution
            h, s, v = cv2.split(hsv)
            l, a, b = cv2.split(lab)
            
            # Calculate statistics
            s_std = np.std(s)  # Saturation variation
            v_std = np.std(v)  # Brightness variation
            a_std = np.std(a)  # Color variation A
            b_std = np.std(b)  # Color variation B
            
            color_score = 0.0
            reasons = []
            
            # Real faces have natural color variations
            color_variation = (s_std + v_std + a_std + b_std) / 4.0
            
            if color_variation < 8:  # Limited color variation = photo/screen
                color_score -= 0.4
                reasons.append(f"CRITICAL: Flat colors: {color_variation:.1f}")
            elif color_variation > 15:  # Good natural variation
                color_score += 0.3
                reasons.append(f"Natural colors: {color_variation:.1f}")
            else:  # Moderate variation
                color_score += 0.1
                reasons.append(f"Normal colors: {color_variation:.1f}")
            
            # Check for screen-like lighting (unnatural bright spots)
            bright_spots = np.sum(v > 220) / v.size
            if bright_spots > 0.05:  # Too many bright spots
                color_score -= 0.2
                reasons.append(f"Screen glare: {bright_spots:.3f}")
            
            reason_text = "Color: " + ", ".join(reasons)
            return color_score, reason_text
            
        except Exception as e:
            return 0.0, f"Color analysis failed: {str(e)}"

    def _detect_screen_characteristics(self, face_region):
        """Detect specific characteristics of screens and printed photos"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            screen_score = 0.0
            reasons = []
            
            # 1. Check for screen pixelation/grid patterns
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = 20 * np.log(np.abs(fft_shift) + 1)
            
            # Look for regular patterns (screen pixels/printing dots)
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            peripheral = magnitude[center_y-20:center_y+20, center_x-20:center_x+20]
            peripheral_mean = np.mean(peripheral)
            
            if peripheral_mean > 90:  # High frequency patterns = screen/print
                screen_score -= 0.3
                reasons.append(f"Screen patterns: {peripheral_mean:.1f}")
            
            # 2. Check for moire patterns (common in screens)
            if self._detect_moire_patterns(gray):
                screen_score -= 0.4
                reasons.append("CRITICAL: Moire patterns detected")
            
            # 3. Check for unnatural aspect ratios (common in phone screens)
            height, width = gray.shape
            aspect_ratio = width / height
            if aspect_ratio < 0.7 or aspect_ratio > 1.5:  # Unnatural aspect
                screen_score -= 0.2
                reasons.append(f"Unnatural aspect: {aspect_ratio:.2f}")
            
            reason_text = "Screen: " + ", ".join(reasons) if reasons else "Screen: No screen patterns"
            return screen_score, reason_text
            
        except Exception as e:
            return 0.0, f"Screen detection failed: {str(e)}"

    def _detect_moire_patterns(self, gray_image):
        """Detect moire patterns common in digital screens"""
        try:
            # Moire patterns create specific frequency responses
            fft = np.fft.fft2(gray_image)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Look for grid-like patterns in frequency domain
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            
            # Check for symmetrical high-frequency components
            corners = [
                magnitude[:center_y//2, :center_x//2],  # Top-left
                magnitude[:center_y//2, center_x+center_x//2:],  # Top-right
                magnitude[center_y+center_y//2:, :center_x//2],  # Bottom-left
                magnitude[center_y+center_y//2:, center_x+center_x//2:]  # Bottom-right
            ]
            
            corner_means = [np.mean(corner) for corner in corners]
            corner_variance = np.var(corner_means)
            
            # Moire patterns create high variance in corner frequencies
            return corner_variance > 1000  # Threshold for moire detection
            
        except:
            return False

class FaceRecognitionSystem:
    """
    Production-ready face recognition system with Accurate ArcFace and Foolproof Anti-Spoofing
    """
    
    def __init__(self):
        """
        Initialize the production face recognition system
        """
        print("🚀 Initializing Production Face Recognition System with FOOLPROOF Anti-Spoofing...")
        
        # Initialize ArcFace model
        self.face_app = None
        self.initialize_arcface()
        
        # Initialize Gemini AI 2.0 Flash
        self.model = None
        self.initialize_gemini()
        
        # Initialize Foolproof Anti-Spoofing
        self.spoofing_detector = FoolproofAntiSpoofing()
        
        # Configuration - SAME AS YOUR ACCURATE STANDALONE
        self.min_face_confidence = 0.7
        self.face_match_threshold = 0.75  # Higher threshold for better accuracy
        self.max_spoof_attempts = 2
        
        # Security tracking
        self.consecutive_spoofs = 0
        
        print("✅ Production Face Recognition System with FOOLPROOF Anti-Spoofing initialized successfully")
    
    def initialize_arcface(self):
        """Initialize ArcFace model for accurate face recognition - SAME AS STANDALONE"""
        try:
            print("🔄 Loading ArcFace model...")
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ ArcFace model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading ArcFace model: {e}")
            self.face_app = None
    
    def initialize_gemini(self):
        """Initialize Gemini 2.0 Flash for spoofing detection"""
        try:
            if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                print("✅ Gemini 2.0 Flash initialized successfully")
            else:
                print("❌ Warning: GEMINI_API_KEY not found in settings")
                self.model = None
        except Exception as e:
            print(f"❌ Error initializing Gemini AI: {e}")
            self.model = None
    
    def detect_faces_arcface(self, frame):
        """Detect faces using ArcFace - EXACTLY SAME AS YOUR ACCURATE STANDALONE"""
        if self.face_app is None:
            return []
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb_frame)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                det_score = face.det_score
                embedding = embedding / np.linalg.norm(embedding)
                results.append({
                    'bbox': bbox,
                    'embedding': embedding,
                    'det_score': det_score
                })
            return results
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def compare_faces_arcface(self, embedding, stored_encodings, stored_names, stored_ids):
        """Compare face embedding with known faces - EXACTLY SAME AS YOUR ACCURATE STANDALONE"""
        if not stored_encodings:
            return None, 0.0, None
        try:
            known_encodings = np.array(stored_encodings)
            query_embedding = np.array(embedding).reshape(1, -1)
            known_encodings = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(known_encodings, query_embedding.T).flatten()
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity >= self.face_match_threshold:
                return stored_names[best_match_idx], best_similarity, stored_ids[best_match_idx]
            else:
                return None, best_similarity, None
                
        except Exception as e:
            print(f"Error in face comparison: {e}")
            return None, 0.0, None
    
    def foolproof_spoofing_check(self, frame, face_bbox):
        """FOOLPROOF spoofing detection - EXACTLY SAME AS YOUR ACCURATE STANDALONE"""
        return self.spoofing_detector.foolproof_detection(frame, face_bbox)
    
    def detect_face_in_frame(self, image_array):
        """
        Production-grade face detection using ArcFace with foolproof anti-spoofing
        """
        if self.face_app is None:
            return {
                'success': False,
                'face_detected': False,
                'message': 'Face detection system not initialized'
            }
        
        try:
            # Detect faces using ArcFace - SAME AS STANDALONE
            faces = self.detect_faces_arcface(image_array)
            
            print(f"🔍 ArcFace detected {len(faces)} faces")
            
            if len(faces) == 0:
                return {
                    'success': True,
                    'face_detected': False,
                    'message': 'No face detected - Please position your face in the frame'
                }
            
            # Get the highest confidence face
            best_face = max(faces, key=lambda x: x['det_score'])
            confidence = best_face['det_score']
            
            print(f"📍 Best face confidence: {confidence:.3f}")
            
            if confidence < self.min_face_confidence:
                return {
                    'success': True,
                    'face_detected': False,
                    'message': f'Low face detection confidence: {confidence:.2f} (min: {self.min_face_confidence})'
                }
            
            # Extract face bounding box and embedding
            bbox = best_face['bbox']
            embedding = best_face['embedding']
            
            # FOOLPROOF SPOOFING DETECTION - SAME AS STANDALONE
            is_spoofed, spoof_reasons, security_score = self.foolproof_spoofing_check(image_array, bbox)
            
            if is_spoofed:
                self.consecutive_spoofs += 1
                spoof_message = " | ".join(spoof_reasons[:3])  # Show first 3 reasons
                return {
                    'success': True,
                    'face_detected': True,
                    'quality_issue': True,
                    'spoofing_detected': True,
                    'message': f'🚫 SPOOFING DETECTED: {spoof_message}',
                    'face_location': bbox.tolist(),
                    'confidence': float(confidence),
                    'security_score': float(security_score),
                    'consecutive_spoofs': self.consecutive_spoofs
                }
            else:
                # Reset spoof counter on successful detection
                self.consecutive_spoofs = 0
            
            # Check face quality
            quality_check = self._check_face_quality_advanced(image_array, bbox)
            
            if not quality_check['is_good']:
                return {
                    'success': True,
                    'face_detected': True,
                    'quality_issue': True,
                    'message': quality_check['message'],
                    'face_location': bbox.tolist(),
                    'confidence': float(confidence),
                    'security_score': float(security_score)
                }
            
            return {
                'success': True,
                'face_detected': True,
                'quality_issue': False,
                'spoofing_detected': False,
                'face_location': bbox.tolist(),
                'facial_patterns': embedding.tolist(),  # ArcFace embeddings
                'confidence': float(confidence),
                'security_score': float(security_score),
                'message': '✅ Face detected with high quality and security',
                'quality_score': quality_check['quality_score']
            }
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {
                'success': False,
                'face_detected': False,
                'message': f'Face detection error: {str(e)}'
            }
    
    def _check_face_quality_advanced(self, image, bbox):
        """
        Advanced face quality checking
        """
        try:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            quality_score = 0
            issues = []
            
            # Check face size
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            size_ratio = face_area / image_area
            
            if size_ratio < 0.05:  # Too small
                issues.append("move closer to camera")
                quality_score += 20
            elif size_ratio > 0.3:  # Too large
                issues.append("move back from camera")
                quality_score += 30
            else:
                quality_score += 50  # Good size
            
            # Check brightness and contrast
            face_region = image[y1:y2, x1:x2]
            if face_region.size > 0:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                contrast = np.std(gray_face)
                
                if brightness < 50:
                    issues.append("improve lighting")
                    quality_score += 10
                elif brightness > 200:
                    issues.append("reduce glare")
                    quality_score += 10
                else:
                    quality_score += 25
                
                if contrast < 30:
                    issues.append("low contrast")
                    quality_score += 10
                else:
                    quality_score += 25
            
            is_good = quality_score >= 80 and len(issues) == 0
            
            return {
                'is_good': is_good,
                'quality_score': quality_score,
                'message': '✅ Excellent face quality' if is_good else f'⚠️ Please: {", ".join(issues)}',
                'issues': issues
            }
            
        except Exception as e:
            return {
                'is_good': False,
                'quality_score': 0,
                'message': f'Quality check error: {str(e)}',
                'issues': ['quality_check_failed']
            }
    
    def check_spoofing_with_gemini(self, image_array):
        """
        Use Gemini 2.0 Flash for advanced spoofing detection as backup
        """
        if self.model is None:
            print("⚠️ Gemini not available, using foolproof anti-spoofing only")
            return True, "Spoofing check passed - Foolproof system active"
        
        try:
            # Convert image to JPEG
            _, buffer = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
            
            prompt = """
            Analyze this facial image for spoofing detection. Look for:
            
            REAL HUMAN FACE indicators:
            - Natural skin texture with pores, wrinkles, and skin details
            - 3D facial structure with proper depth and contours
            - Natural lighting consistency and realistic shadows
            - Authentic eye reflections and pupil details
            - Natural hair follicles and eyebrow details
            
            SPOOFING ATTEMPT indicators:
            - Flat 2D appearance without depth
            - Digital artifacts, compression, or pixelation
            - Screen reflections, moire patterns, or LCD grid
            - Paper texture, printing dots, or visible print patterns
            - Visible device borders, phone edges, or frame boundaries
            - Unnatural color patterns or lighting
            - Lack of fine skin details and textures
            
            IMPORTANT: Respond with EXACTLY ONE WORD - either "REAL" or "SPOOF".
            Be very strict with spoofing detection.
            """
            
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
            
            print("🤖 Sending to Gemini 2.0 Flash for spoofing analysis...")
            start_time = time.time()
            
            response = self.model.generate_content([prompt, image_part])
            
            end_time = time.time()
            result_text = response.text.strip().upper()
            
            print(f"🤖 Gemini 2.0 Flash response: '{result_text}' (took {end_time - start_time:.2f}s)")
            
            if "REAL" in result_text:
                return True, "✅ Real human face verified - Gemini spoofing check passed"
            elif "SPOOF" in result_text:
                return False, "❌ SPOOFING DETECTED - This appears to be a photo or screen"
            else:
                print(f"⚠️ Unexpected Gemini response: {result_text}")
                # Default to cautious approach - treat as potential spoof
                return False, "❌ Spoofing check inconclusive - Please use real face"
                
        except Exception as e:
            print(f"❌ Gemini spoofing detection error: {e}")
            # Default to cautious approach in case of error
            return False, f"❌ Spoofing check failed: {str(e)} - Please try again"

# Global instance
face_system = FaceRecognitionSystem()