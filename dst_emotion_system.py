import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import hashlib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class DSTEnhancedEmotionDetectionSystem:
    """
    DST-Enhanced Facial Emotion Detection System (No TensorFlow Required)
    
    Primary Purpose: Facial Emotion Detection (happy, sad, neutral, surprise)
    Enhancement: DST-based uncertainty quantification when confidence is low
    Uses: scikit-learn, OpenCV, traditional ML approaches
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.emotion_classes = ['happy', 'sad', 'neutral', 'surprise']
        self.img_size = 48
        
        # Core emotion detection models
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=100)
        
        # DST enhancement parameters
        self.uncertainty_threshold = 0.3
        self.dst_active = False
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        self.dst_enhancement_stats = {}
        
        # Data quality assessment
        self.data_quality_results = {}
        self.duplicate_mapping = defaultdict(list)
        self.face_cascade = None
        self._initialize_face_detector()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _initialize_face_detector(self):
        """
        Initialize OpenCV face detection cascade
        """
        try:
            # Try to load the face cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("⚠️  Face detection cascade not loaded properly")
                self.face_cascade = None
        except:
            print("⚠️  Face detection not available")
            self.face_cascade = None
    
    def _compute_perceptual_hash(self, image_path):
        """
        Compute perceptual hash for duplicate detection
        """
        try:
            # Load image with PIL for consistent processing
            with Image.open(image_path) as img:
                # Convert to grayscale and resize for consistent hashing
                img = img.convert('L').resize((16, 16), Image.Resampling.LANCZOS)
                # Convert to numpy array
                img_array = np.array(img)
                # Compute average hash
                avg = np.mean(img_array)
                # Create binary hash
                hash_bits = (img_array > avg).flatten()
                # Convert to string for easy storage and comparison
                hash_string = ''.join(['1' if bit else '0' for bit in hash_bits])
                return hash_string
        except Exception as e:
            return None
    
    def _validate_image_content(self, image_path):
        """
        Validate image content for emotion labeling appropriateness
        """
        validation_results = {
            'is_valid_file': False,
            'is_loadable': False,
            'has_content': False,
            'dimensions_valid': False,
            'has_faces': False,
            'face_count': 0,
            'is_corrupted': False,
            'content_type': 'unknown',
            'issues': []
        }
        
        try:
            # Check if file exists and is readable
            if not image_path.exists():
                validation_results['issues'].append('file_not_found')
                return validation_results
                
            validation_results['is_valid_file'] = True
            
            # Try to load with OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                validation_results['issues'].append('opencv_load_failed')
                return validation_results
                
            validation_results['is_loadable'] = True
            
            # Check image dimensions
            height, width = img.shape[:2]
            if height < 10 or width < 10:
                validation_results['issues'].append('dimensions_too_small')
                validation_results['dimensions_valid'] = False
            else:
                validation_results['dimensions_valid'] = True
            
            # Check if image has meaningful content (not all black, all white, etc.)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check for all black/white images
            if np.all(gray == 0):
                validation_results['issues'].append('all_black_image')
                validation_results['content_type'] = 'all_black'
            elif np.all(gray == 255):
                validation_results['issues'].append('all_white_image')
                validation_results['content_type'] = 'all_white'
            elif np.std(gray) < 5:  # Very low variance indicates uniform color
                validation_results['issues'].append('uniform_color')
                validation_results['content_type'] = 'uniform'
            else:
                validation_results['has_content'] = True
                validation_results['content_type'] = 'normal'
            
            # Face detection if available
            if self.face_cascade is not None and validation_results['has_content']:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                validation_results['face_count'] = len(faces)
                validation_results['has_faces'] = len(faces) > 0
                
                if len(faces) == 0:
                    validation_results['issues'].append('no_faces_detected')
                elif len(faces) > 3:
                    validation_results['issues'].append('too_many_faces')
            
        except Exception as e:
            validation_results['issues'].append(f'validation_error: {str(e)}')
            validation_results['is_corrupted'] = True
        
        return validation_results
    
    def perform_comprehensive_data_quality_assessment(self):
        """
        Perform comprehensive data quality assessment on the entire dataset
        """
        print(f"🔍 Performing Comprehensive Data Quality Assessment...")
        
        quality_stats = {
            'total_files_found': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'no_face_files': 0,
            'duplicate_groups': 0,
            'total_duplicates': 0,
            'cross_emotion_duplicates': 0,
            'cross_split_duplicates': 0,
            'problematic_files': [],
            'duplicate_clusters': [],
            'file_issues': defaultdict(int)
        }
        
        # Hash mapping for duplicate detection
        hash_to_files = defaultdict(list)
        file_validation_results = {}
        
        # Process all files in dataset
        for split in ['train', 'validation']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            for emotion in self.emotion_classes:
                emotion_path = split_path / emotion
                if not emotion_path.exists():
                    continue
                    
                image_files = list(emotion_path.glob('*.jpg')) + list(emotion_path.glob('*.png'))
                quality_stats['total_files_found'] += len(image_files)
                
                print(f"🔍 Analyzing {split}/{emotion}: {len(image_files)} files")
                
                for img_path in image_files:
                    # Compute perceptual hash for duplicate detection
                    img_hash = self._compute_perceptual_hash(img_path)
                    if img_hash:
                        file_info = {
                            'path': img_path,
                            'split': split,
                            'emotion': emotion,
                            'filename': img_path.name
                        }
                        hash_to_files[img_hash].append(file_info)
                    
                    # Validate image content
                    validation = self._validate_image_content(img_path)
                    file_validation_results[str(img_path)] = validation
                    
                    # Update statistics
                    if validation['is_loadable'] and validation['has_content']:
                        quality_stats['valid_files'] += 1
                    else:
                        quality_stats['corrupted_files'] += 1
                        quality_stats['problematic_files'].append({
                            'path': str(img_path),
                            'split': split,
                            'emotion': emotion,
                            'issues': validation['issues']
                        })
                    
                    if not validation['has_faces']:
                        quality_stats['no_face_files'] += 1
                    
                    # Count issue types
                    for issue in validation['issues']:
                        quality_stats['file_issues'][issue] += 1
        
        # Analyze duplicates
        print(f"🔍 Analyzing duplicate patterns...")
        for img_hash, file_list in hash_to_files.items():
            if len(file_list) > 1:
                quality_stats['duplicate_groups'] += 1
                quality_stats['total_duplicates'] += len(file_list)
                
                # Check for cross-emotion duplicates
                emotions = set(f['emotion'] for f in file_list)
                if len(emotions) > 1:
                    quality_stats['cross_emotion_duplicates'] += 1
                
                # Check for cross-split duplicates (train/validation contamination)
                splits = set(f['split'] for f in file_list)
                if len(splits) > 1:
                    quality_stats['cross_split_duplicates'] += 1
                
                # Store duplicate cluster
                quality_stats['duplicate_clusters'].append({
                    'hash': img_hash,
                    'files': file_list,
                    'count': len(file_list),
                    'cross_emotion': len(emotions) > 1,
                    'cross_split': len(splits) > 1,
                    'emotions': list(emotions),
                    'splits': list(splits)
                })
        
        self.data_quality_results = {
            'statistics': quality_stats,
            'file_validations': file_validation_results,
            'hash_mapping': hash_to_files
        }
        
        # Print summary
        print(f"📊 Data Quality Assessment Complete:")
        print(f"   Total files found: {quality_stats['total_files_found']}")
        print(f"   Valid files: {quality_stats['valid_files']}")
        print(f"   Corrupted/Invalid files: {quality_stats['corrupted_files']}")
        print(f"   Files with no faces: {quality_stats['no_face_files']}")
        print(f"   Duplicate groups found: {quality_stats['duplicate_groups']}")
        print(f"   Total duplicate instances: {quality_stats['total_duplicates']}")
        print(f"   Cross-emotion duplicates: {quality_stats['cross_emotion_duplicates']}")
        print(f"   Cross-split contamination: {quality_stats['cross_split_duplicates']}")
        
        if quality_stats['cross_emotion_duplicates'] > 0 or quality_stats['cross_split_duplicates'] > 0:
            print(f"⚠️  CRITICAL: Dataset contamination detected!")
        
        return self.data_quality_results
    
    def flag_specific_duplicates_for_review(self, target_filenames=None):
        """
        Flag specific duplicates for human review (your evidence files)
        """
        if target_filenames is None:
            target_filenames = ['happy_08_26383_score29.9.jpg', 'happy_2102_29.9.jpg']
        
        flagged_files = []
        
        if not hasattr(self, 'data_quality_results') or not self.data_quality_results:
            print("⚠️  Data quality assessment not performed. Run perform_comprehensive_data_quality_assessment() first.")
            return flagged_files
        
        # Search for target files in duplicate clusters
        for cluster in self.data_quality_results['statistics']['duplicate_clusters']:
            cluster_filenames = [f['filename'] for f in cluster['files']]
            
            # Check if any target files are in this cluster
            found_targets = [name for name in target_filenames if name in cluster_filenames]
            
            if found_targets:
                flagged_files.append({
                    'cluster_info': cluster,
                    'target_files_found': found_targets,
                    'evidence_type': 'dataset_contamination',
                    'recommendation': 'human_review_required',
                    'academic_significance': 'proves_data_quality_issues'
                })
                
                print(f"🚨 EVIDENCE FLAGGED: Found target duplicate cluster!")
                print(f"   Target files: {found_targets}")
                print(f"   All files in cluster: {cluster_filenames}")
                print(f"   Cross-emotion: {cluster['cross_emotion']}")
                print(f"   Cross-split: {cluster['cross_split']}")
        
        return flagged_files
    
    def extract_image_features(self, image_path):
        """
        Extract comprehensive features from facial images
        """
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            gray = cv2.resize(gray, (self.img_size, self.img_size))
            
            # Feature extraction methods
            features = []
            
            # 1. Raw pixel features (flattened)
            pixel_features = gray.flatten() / 255.0
            features.extend(pixel_features)
            
            # 2. Histogram features
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist_features = hist.flatten() / np.sum(hist)
            features.extend(hist_features)
            
            # 3. LBP (Local Binary Pattern) features
            try:
                lbp_features = self._compute_lbp(gray)
                features.extend(lbp_features)
            except:
                # Fallback if LBP computation fails
                features.extend([0] * 58)  # 58 LBP features
            
            # 4. Statistical features
            stat_features = [
                np.mean(gray), np.std(gray), np.median(gray),
                np.min(gray), np.max(gray), np.percentile(gray, 25),
                np.percentile(gray, 75), np.var(gray)
            ]
            features.extend(stat_features)
            
            # 5. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_features = [
                np.sum(edges > 0) / (self.img_size * self.img_size),  # Edge density
                np.mean(edges), np.std(edges)
            ]
            features.extend(edge_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def _compute_lbp(self, image):
        """
        Compute Local Binary Pattern features
        """
        # Simple LBP implementation
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Compute histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=58, range=(0, 58))
        return hist / np.sum(hist)
    
    def load_and_prepare_data(self):
        """
        Load images and extract features for training - PROCESSES ALL FILES
        """
        print(f"📊 Loading and preparing emotion detection data...")
        print(f"🎭 Target emotions: {self.emotion_classes}")
        print(f"💾 Processing ALL available files (no artificial limits)")
        
        all_features = []
        all_labels = []
        
        # Load data from train and validation folders
        for split in ['train', 'validation']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            for emotion in self.emotion_classes:
                emotion_path = split_path / emotion
                if not emotion_path.exists():
                    continue
                    
                image_files = list(emotion_path.glob('*.jpg')) + list(emotion_path.glob('*.png'))
                print(f"Processing {split}/{emotion}: {len(image_files)} images")
                
                # REMOVED ARTIFICIAL LIMITS - Process all files
                for img_path in image_files:
                    features = self.extract_image_features(img_path)
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(emotion)
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"✅ Data loaded successfully")
        print(f"📊 Total samples: {len(X)}")
        print(f"📏 Feature dimension: {X.shape[1]}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Apply PCA for dimensionality reduction
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        print(f"📈 Training samples: {len(self.X_train)}")
        print(f"🧪 Test samples: {len(self.X_test)}")
        print(f"🔍 PCA components: {self.pca.n_components_}")
        
        return self.X_train_pca, self.X_test_pca, self.y_train, self.y_test
    
    def build_emotion_detection_model(self):
        """
        Build ensemble emotion detection model using scikit-learn
        """
        print(f"🧠 Building ensemble emotion detection model...")
        
        # Create individual models
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        svm_model = SVC(
            kernel='rbf',
            probability=True,  # Important for uncertainty estimation
            random_state=42
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
        
        # Create ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('svm', svm_model),
                ('knn', knn_model)
            ],
            voting='soft'  # Use probabilities for DST enhancement
        )
        
        print(f"✅ Ensemble model built with 3 base classifiers")
        return self.ensemble_model
    
    def train_emotion_model(self):
        """
        Train the emotion detection model
        """
        print(f"🚀 Training emotion detection model...")
        
        if self.ensemble_model is None:
            raise ValueError("Model not built. Call build_emotion_detection_model() first.")
        
        # Train the ensemble
        self.ensemble_model.fit(self.X_train_pca, self.y_train)
        
        # Evaluate training performance
        train_predictions = self.ensemble_model.predict(self.X_train_pca)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble_model, self.X_train_pca, self.y_train, cv=5)
        
        self.training_results = {
            'train_accuracy': train_accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }
        
        print(f"✅ Training completed!")
        print(f"📊 Training Accuracy: {train_accuracy:.3f}")
        print(f"📊 Cross-Validation Mean: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
        
        return self.training_results
    
    def evaluate_emotion_model(self):
        """
        Evaluate the emotion detection model on test set
        """
        print(f"🧪 Evaluating emotion detection model on test set...")
        
        # Predictions
        test_predictions = self.ensemble_model.predict(self.X_test_pca)
        test_probabilities = self.ensemble_model.predict_proba(self.X_test_pca)
        
        # Accuracy
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        
        # Detailed classification report
        emotion_names = self.label_encoder.classes_
        class_report = classification_report(
            self.y_test, 
            test_predictions, 
            target_names=emotion_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, test_predictions)
        
        self.evaluation_results = {
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': test_predictions,
            'probabilities': test_probabilities,
            'true_labels': self.y_test,
            'emotion_names': emotion_names
        }
        
        print(f"📊 Test Accuracy: {test_accuracy:.3f}")
        
        # Print detailed results
        print(f"\n📋 Classification Report:")
        print(classification_report(self.y_test, test_predictions, target_names=emotion_names))
        
        return self.evaluation_results
    
    def detect_prediction_uncertainty(self, probabilities):
        """
        Detect when DST uncertainty quantification should be triggered
        """
        uncertainties = []
        
        for prob in probabilities:
            # Calculate prediction confidence (max probability)
            max_confidence = np.max(prob)
            
            # Calculate entropy-based uncertainty
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            normalized_entropy = entropy / np.log(len(prob))
            
            # Calculate prediction margin (difference between top 2 predictions)
            sorted_prob = np.sort(prob)[::-1]
            margin = sorted_prob[0] - sorted_prob[1]
            
            # Combined uncertainty score
            uncertainty_score = (normalized_entropy * 0.5) + ((1 - max_confidence) * 0.3) + ((1 - margin) * 0.2)
            
            uncertainties.append({
                'max_confidence': max_confidence,
                'entropy': normalized_entropy,
                'margin': margin,
                'uncertainty_score': uncertainty_score,
                'needs_dst_analysis': uncertainty_score > self.uncertainty_threshold
            })
        
        return uncertainties
    
    def dst_uncertainty_analysis(self, uncertain_probabilities, uncertain_indices):
        """
        DST-based uncertainty analysis for cases flagged as uncertain
        """
        print(f"🧠 DST Analysis triggered for {len(uncertain_indices)} uncertain predictions")
        
        dst_results = []
        emotion_names = self.evaluation_results['emotion_names']
        
        for i, (pred_idx, probability) in enumerate(zip(uncertain_indices, uncertain_probabilities)):
            # Get the original prediction probabilities
            emotion_probs = probability
            predicted_emotion_idx = np.argmax(emotion_probs)
            predicted_emotion = emotion_names[predicted_emotion_idx]
            max_prob = np.max(emotion_probs)
            
            # DST Evidence Sources
            evidence_sources = {}
            
            # Evidence 1: Ensemble agreement analysis
            try:
                individual_probs = []
                for estimator_name, estimator in self.ensemble_model.named_estimators_.items():
                    individual_prob = estimator.predict_proba(self.X_test_pca[pred_idx:pred_idx+1])[0]
                    individual_probs.append(individual_prob)
                
                # Calculate agreement between models
                predictions = [np.argmax(prob) for prob in individual_probs]
                agreement = len(set(predictions)) == 1  # All models agree
                
                evidence_sources['ensemble_agreement'] = {
                    'agreement': agreement,
                    'individual_predictions': predictions,
                    'reliability': 0.9 if agreement else 0.4
                }
            except:
                evidence_sources['ensemble_agreement'] = {
                    'agreement': True,
                    'reliability': 0.7
                }
            
            # Evidence 2: Probability distribution analysis
            entropy = -np.sum(emotion_probs * np.log(emotion_probs + 1e-10))
            normalized_entropy = entropy / np.log(len(emotion_probs))
            
            evidence_sources['probability_distribution'] = {
                'entropy': normalized_entropy,
                'uniformity': normalized_entropy,
                'reliability': 1.0 - normalized_entropy
            }
            
            # Evidence 3: Top-k prediction analysis
            sorted_indices = np.argsort(emotion_probs)[::-1]
            top_2_probs = emotion_probs[sorted_indices[:2]]
            margin = top_2_probs[0] - top_2_probs[1]
            
            evidence_sources['top_predictions'] = {
                'margin': margin,
                'top_emotion': emotion_names[sorted_indices[0]],
                'second_emotion': emotion_names[sorted_indices[1]],
                'reliability': margin
            }
            
            # DST Combination
            combined_confidence = self._dst_combine_evidence(evidence_sources)
            
            # Generate explanation
            explanation = self._generate_uncertainty_explanation(
                predicted_emotion, combined_confidence, evidence_sources
            )
            
            dst_result = {
                'prediction_index': pred_idx,
                'original_prediction': predicted_emotion,
                'original_confidence': max_prob,
                'dst_combined_confidence': combined_confidence['final_confidence'],
                'dst_uncertainty': combined_confidence['uncertainty'],
                'evidence_sources': evidence_sources,
                'explanation': explanation,
                'recommendation': self._get_dst_recommendation(combined_confidence)
            }
            
            dst_results.append(dst_result)
        
        return dst_results
    
    def _dst_combine_evidence(self, evidence_sources):
        """
        Simplified DST evidence combination
        """
        weights = {
            'ensemble_agreement': 0.4,
            'probability_distribution': 0.35,
            'top_predictions': 0.25
        }
        
        total_confidence = 0
        total_weight = 0
        
        for source, data in evidence_sources.items():
            if source in weights:
                reliability = data['reliability']
                weight = weights[source] * reliability
                total_confidence += weight * reliability
                total_weight += weight
        
        if total_weight > 0:
            final_confidence = total_confidence / total_weight
        else:
            final_confidence = 0.5
        
        uncertainty = 1.0 - final_confidence
        
        return {
            'final_confidence': final_confidence,
            'uncertainty': uncertainty,
            'total_weight': total_weight
        }
    
    def _generate_uncertainty_explanation(self, predicted_emotion, combined_confidence, evidence_sources):
        """
        Generate human-readable explanation
        """
        conf = combined_confidence['final_confidence']
        uncertainty = combined_confidence['uncertainty']
        
        if uncertainty > 0.7:
            return f"High uncertainty for '{predicted_emotion}' prediction. Multiple emotions possible or expression ambiguous. Human review recommended."
        elif uncertainty > 0.4:
            return f"Moderate uncertainty for '{predicted_emotion}' prediction. Confidence: {conf:.2f}. Consider additional context."
        else:
            return f"Low uncertainty for '{predicted_emotion}' prediction. DST confirms confidence: {conf:.2f}."
    
    def _get_dst_recommendation(self, combined_confidence):
        """
        Get recommendation based on DST analysis
        """
        uncertainty = combined_confidence['uncertainty']
        
        if uncertainty > 0.7:
            return "human_review_strongly_recommended"
        elif uncertainty > 0.4:
            return "consider_additional_context"
        else:
            return "proceed_with_caution"
    
    def analyze_with_dst_enhancement(self):
        """
        Perform emotion detection with DST enhancement for uncertain cases
        """
        print(f"🎭 Running Emotion Detection with DST Enhancement...")
        
        if not self.evaluation_results:
            raise ValueError("Model not evaluated. Run evaluate_emotion_model() first.")
        
        # Get predictions and uncertainties
        probabilities = self.evaluation_results['probabilities']
        uncertainties = self.detect_prediction_uncertainty(probabilities)
        
        # Identify uncertain cases
        uncertain_indices = [i for i, u in enumerate(uncertainties) if u['needs_dst_analysis']]
        uncertain_probabilities = [probabilities[i] for i in uncertain_indices]
        
        print(f"📊 Total predictions: {len(probabilities)}")
        print(f"⚠️  Uncertain cases detected: {len(uncertain_indices)} ({len(uncertain_indices)/len(probabilities)*100:.1f}%)")
        
        # Trigger DST analysis only for uncertain cases
        if len(uncertain_indices) > 0:
            self.dst_active = True
            dst_results = self.dst_uncertainty_analysis(uncertain_probabilities, uncertain_indices)
            
            # Analyze DST results
            dst_stats = self._analyze_dst_results(dst_results)
            
            self.dst_enhancement_stats = {
                'total_predictions': len(probabilities),
                'uncertain_cases': len(uncertain_indices),
                'uncertainty_rate': len(uncertain_indices) / len(probabilities),
                'dst_results': dst_results,
                'dst_statistics': dst_stats
            }
            
            print(f"🧠 DST Analysis Complete:")
            print(f"   High uncertainty cases: {dst_stats['high_uncertainty']}")
            print(f"   Moderate uncertainty: {dst_stats['moderate_uncertainty']}")
            print(f"   Confirmed predictions: {dst_stats['low_uncertainty']}")
            print(f"   Human review recommended: {dst_stats['human_review_recommended']}")
            
        else:
            print(f"✅ No uncertain cases detected - DST enhancement not needed!")
            self.dst_enhancement_stats = {
                'total_predictions': len(probabilities),
                'uncertain_cases': 0,
                'uncertainty_rate': 0,
                'message': 'All predictions had sufficient confidence - DST not triggered'
            }
        
        return self.dst_enhancement_stats
    
    def _analyze_dst_results(self, dst_results):
        """
        Analyze DST enhancement results
        """
        stats = {
            'high_uncertainty': 0,
            'moderate_uncertainty': 0,
            'low_uncertainty': 0,
            'human_review_recommended': 0
        }
        
        for result in dst_results:
            uncertainty = result['dst_uncertainty']
            
            if uncertainty > 0.7:
                stats['high_uncertainty'] += 1
            elif uncertainty > 0.4:
                stats['moderate_uncertainty'] += 1
            else:
                stats['low_uncertainty'] += 1
            
            if result['recommendation'] == 'human_review_strongly_recommended':
                stats['human_review_recommended'] += 1
        
        return stats
    
    def create_comprehensive_visualizations(self, save_path=r'C:\\projects\\AI\\sklearn_emotion_results.png'):
        """
        Create comprehensive visualizations including data quality metrics
        """
        if self.dst_active:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('DST-Enhanced Emotion Detection with Data Quality Assessment', fontsize=16, fontweight='bold')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Emotion Detection Results with Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        emotion_names = self.evaluation_results['emotion_names']
        sns.heatmap(self.evaluation_results['confusion_matrix'], 
                   annot=True, fmt='d', ax=ax1,
                   xticklabels=emotion_names,
                   yticklabels=emotion_names)
        ax1.set_title('Emotion Detection Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Per-Class Performance
        ax2 = axes[0, 1]
        class_report = self.evaluation_results['classification_report']
        f1_scores = [class_report[emotion]['f1-score'] for emotion in emotion_names]
        
        bars = ax2.bar(emotion_names, f1_scores, color='skyblue', alpha=0.7)
        ax2.set_title('Per-Emotion F1 Scores')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        
        for bar, score in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Additional plots based on DST activation and data quality results
        if hasattr(self, 'data_quality_results') and self.data_quality_results:
            # Data Quality Overview
            if self.dst_active:
                ax3 = axes[0, 2]
            else:
                ax3 = axes[1, 0]
            
            quality_stats = self.data_quality_results['statistics']
            categories = ['Valid Files', 'Corrupted', 'No Faces', 'Duplicates']
            values = [
                quality_stats['valid_files'],
                quality_stats['corrupted_files'], 
                quality_stats['no_face_files'],
                quality_stats['total_duplicates']
            ]
            
            bars = ax3.bar(categories, values, color=['green', 'red', 'orange', 'purple'], alpha=0.7)
            ax3.set_title('Data Quality Assessment')
            ax3.set_ylabel('Number of Files')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(value), ha='center', va='bottom')
        
        if self.dst_active:
            # DST Enhancement Impact
            ax4 = axes[1, 0]
            dst_stats = self.dst_enhancement_stats['dst_statistics']
            categories = ['High Uncertainty', 'Moderate', 'Low', 'Human Review']
            values = [dst_stats['high_uncertainty'], dst_stats['moderate_uncertainty'], 
                     dst_stats['low_uncertainty'], dst_stats['human_review_recommended']]
            
            bars = ax4.bar(categories, values, color=['red', 'orange', 'green', 'purple'], alpha=0.7)
            ax4.set_title('DST Uncertainty Analysis Results')
            ax4.set_ylabel('Number of Cases')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(value), ha='center', va='bottom')
            
            # System Performance Overview
            ax5 = axes[1, 1]
            total_preds = self.dst_enhancement_stats['total_predictions']
            uncertain_cases = self.dst_enhancement_stats['uncertain_cases']
            confident_cases = total_preds - uncertain_cases
            
            pie_data = [confident_cases, uncertain_cases]
            pie_labels = [f'Confident\n({confident_cases})', f'DST Enhanced\n({uncertain_cases})']
            colors = ['lightgreen', 'lightcoral']
            
            ax5.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=colors)
            ax5.set_title('DST Enhancement Trigger Rate')
            
            # Model Confidence vs Accuracy
            ax6 = axes[1, 2]
            probabilities = self.evaluation_results['probabilities']
            predictions = self.evaluation_results['predictions']
            true_labels = self.evaluation_results['true_labels']
            
            max_confidences = np.max(probabilities, axis=1)
            accuracies = (predictions == true_labels).astype(int)
            
            # Create confidence bins
            conf_bins = np.linspace(0, 1, 11)
            bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
            bin_accuracies = []
            
            for i in range(len(conf_bins) - 1):
                mask = (max_confidences >= conf_bins[i]) & (max_confidences < conf_bins[i+1])
                if np.sum(mask) > 0:
                    bin_acc = np.mean(accuracies[mask])
                    bin_accuracies.append(bin_acc)
                else:
                    bin_accuracies.append(0)
            
            ax6.plot(bin_centers, bin_accuracies, 'bo-', label='Actual')
            ax6.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            ax6.set_xlabel('Confidence')
            ax6.set_ylabel('Accuracy')
            ax6.set_title('Confidence Calibration')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        else:
            # Model Confidence Distribution (when DST not active)
            ax4 = axes[1, 1]
            probabilities = self.evaluation_results['probabilities']
            max_confidences = np.max(probabilities, axis=1)
            
            ax4.hist(max_confidences, bins=20, alpha=0.7, edgecolor='black', color='lightblue')
            ax4.set_title('Model Confidence Distribution')
            ax4.set_xlabel('Maximum Prediction Confidence')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📊 Visualizations saved to {save_path}")
    
    def generate_comprehensive_report(self, save_path=r'C:\\projects\\AI\\sklearn_emotion_report.txt'):
        """
        Generate comprehensive report including data quality assessment
        """
        report = []
        report.append("=" * 80)
        report.append("DST-ENHANCED EMOTION DETECTION SYSTEM REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Overview
        report.append("🎭 SYSTEM OVERVIEW:")
        report.append(f"   Primary Mission: Facial Emotion Detection")
        report.append(f"   Target Emotions: {', '.join(self.emotion_classes)}")
        report.append(f"   ML Framework: Scikit-Learn Ensemble")
        report.append(f"   Enhancement: DST Uncertainty Quantification")
        report.append(f"   Data Quality: Comprehensive Assessment with Duplicate Detection")
        report.append("")
        
        # Data Quality Results
        if hasattr(self, 'data_quality_results') and self.data_quality_results:
            report.append("🔍 DATA QUALITY ASSESSMENT:")
            quality_stats = self.data_quality_results['statistics']
            report.append(f"   Total files found: {quality_stats['total_files_found']}")
            report.append(f"   Valid files: {quality_stats['valid_files']}")
            report.append(f"   Corrupted/Invalid files: {quality_stats['corrupted_files']}")
            report.append(f"   Files with no faces: {quality_stats['no_face_files']}")
            report.append(f"   Duplicate groups found: {quality_stats['duplicate_groups']}")
            report.append(f"   Total duplicate instances: {quality_stats['total_duplicates']}")
            report.append(f"   Cross-emotion duplicates: {quality_stats['cross_emotion_duplicates']}")
            report.append(f"   Cross-split contamination: {quality_stats['cross_split_duplicates']}")
            
            if quality_stats['cross_emotion_duplicates'] > 0 or quality_stats['cross_split_duplicates'] > 0:
                report.append(f"   ⚠️  CRITICAL: Dataset contamination detected!")
            report.append("")
        
        # Model Architecture
        report.append("🧠 MODEL ARCHITECTURE:")
        report.append(f"   Base Models: Random Forest, SVM, K-Nearest Neighbors")
        report.append(f"   Ensemble Method: Soft Voting Classifier")
        report.append(f"   Feature Engineering: Pixel + Histogram + LBP + Statistical + Edge")
        report.append(f"   Dimensionality Reduction: PCA ({self.pca.n_components_} components)")
        report.append("")
        
        # Training Results
        report.append("📈 TRAINING PERFORMANCE:")
        report.append(f"   Training Accuracy: {self.training_results['train_accuracy']:.3f}")
        report.append(f"   Cross-Validation Mean: {self.training_results['cv_mean']:.3f}")
        report.append(f"   Cross-Validation Std: {self.training_results['cv_std']:.3f}")
        report.append("")
        
        # Test Performance
        report.append("🧪 TEST PERFORMANCE:")
        report.append(f"   Test Accuracy: {self.evaluation_results['test_accuracy']:.3f}")
        report.append("")
        
        # Per-Emotion Performance
        report.append("🎭 PER-EMOTION PERFORMANCE:")
        class_report = self.evaluation_results['classification_report']
        emotion_names = self.evaluation_results['emotion_names']
        for emotion in emotion_names:
            if emotion in class_report:
                precision = class_report[emotion]['precision']
                recall = class_report[emotion]['recall']
                f1 = class_report[emotion]['f1-score']
                report.append(f"   {emotion.capitalize()}:")
                report.append(f"     Precision: {precision:.3f}")
                report.append(f"     Recall: {recall:.3f}")
                report.append(f"     F1-Score: {f1:.3f}")
        report.append("")
        
        # DST Enhancement Results
        if self.dst_active:
            report.append("🧠 DST ENHANCEMENT RESULTS:")
            dst_stats = self.dst_enhancement_stats
            report.append(f"   Total Predictions: {dst_stats['total_predictions']}")
            report.append(f"   Uncertain Cases: {dst_stats['uncertain_cases']}")
            report.append(f"   DST Trigger Rate: {dst_stats['uncertainty_rate']:.1%}")
            report.append("")
            
            if 'dst_statistics' in dst_stats:
                stats = dst_stats['dst_statistics']
                report.append("   DST Analysis Breakdown:")
                report.append(f"     High Uncertainty: {stats['high_uncertainty']}")
                report.append(f"     Moderate Uncertainty: {stats['moderate_uncertainty']}")
                report.append(f"     Low Uncertainty: {stats['low_uncertainty']}")
                report.append(f"     Human Review Recommended: {stats['human_review_recommended']}")
                report.append("")
        else:
            report.append("✅ DST ENHANCEMENT STATUS:")
            report.append("   DST not triggered - all predictions had sufficient confidence")
            report.append("   System operated in standard emotion detection mode")
            report.append("")
        
        # System Advantages
        report.append("🚀 SYSTEM ADVANTAGES:")
        report.append("   ✓ No TensorFlow dependency - runs on any Python environment")
        report.append("   ✓ Ensemble approach provides robust predictions")
        report.append("   ✓ DST enhancement only activates when needed")
        report.append("   ✓ Comprehensive feature engineering")
        report.append("   ✓ Uncertainty quantification for ambiguous cases")
        report.append("   ✓ Advanced data quality assessment with duplicate detection")
        report.append("   ✓ Automatic flagging of problematic samples for human review")
        report.append("")
        
        report.append("=" * 80)
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Comprehensive report saved: {save_path}")
        return '\n'.join(report)

def run_sklearn_emotion_detection_system():
    """
    Run the complete scikit-learn based emotion detection system with data quality assessment
    """
    dataset_path = r"C:\\projects\\AI\\Facial_emotion_images"
    
    print("🎭 Initializing Scikit-Learn DST-Enhanced Emotion Detection System")
    print("="*70)
    
    # Initialize system
    system = DSTEnhancedEmotionDetectionSystem(dataset_path)
    
    # Perform comprehensive data quality assessment first
    print("\n🔍 PHASE 1: Data Quality Assessment")
    print("-" * 50)
    quality_results = system.perform_comprehensive_data_quality_assessment()
    
    # Flag specific duplicates for evidence
    print("\n🚨 PHASE 2: Evidence Detection")
    print("-" * 50)
    flagged_files = system.flag_specific_duplicates_for_review()
    if flagged_files:
        print(f"✅ Found {len(flagged_files)} evidence clusters of dataset contamination")
    else:
        print("ℹ️  No target evidence files found, but comprehensive analysis completed")
    
    # Load and prepare data (now processes ALL files)
    print("\n💾 PHASE 3: Data Loading and Processing")
    print("-" * 50)
    X_train, X_test, y_train, y_test = system.load_and_prepare_data()
    
    # Build model
    print("\n🧠 PHASE 4: Model Building")
    print("-" * 50)
    model = system.build_emotion_detection_model()
    
    # Train model
    print("\n🚀 PHASE 5: Model Training")
    print("-" * 50)
    training_results = system.train_emotion_model()
    
    # Evaluate model
    print("\n🧪 PHASE 6: Model Evaluation")
    print("-" * 50)
    eval_results = system.evaluate_emotion_model()
    
    # DST enhancement analysis
    print("\n🎭 PHASE 7: DST Enhancement Analysis")
    print("-" * 50)
    dst_results = system.analyze_with_dst_enhancement()
    
    # Generate visualizations
    print("\n📊 PHASE 8: Generating Visualizations")
    print("-" * 50)
    system.create_comprehensive_visualizations()
    
    # Generate report
    print("\n📄 PHASE 9: Generating Comprehensive Report")
    print("-" * 50)
    report = system.generate_comprehensive_report()
    
    print("\n" + "="*70)
    print("🎯 ENHANCED EMOTION DETECTION SYSTEM COMPLETE!")
    print("="*70)
    print("🎭 Core Mission: Emotion Detection ✅")
    print("🧠 Enhancement: DST Uncertainty Quantification ✅")
    print("🔍 Data Quality: Comprehensive Assessment ✅")
    print("🚨 Evidence Detection: Dataset Contamination Analysis ✅")
    print("💻 Framework: Scikit-Learn (No TensorFlow) ✅")
    print("📊 Ready for Academic Presentation ✅")
    
    # Summary of findings
    if hasattr(system, 'data_quality_results') and system.data_quality_results:
        quality_stats = system.data_quality_results['statistics']
        print(f"\n📋 KEY FINDINGS:")
        print(f"   📁 Total files processed: {quality_stats['total_files_found']:,}")
        print(f"   ✅ Valid samples: {quality_stats['valid_files']:,}")
        print(f"   ❌ Problematic files: {quality_stats['corrupted_files']:,}")
        print(f"   👤 No face detected: {quality_stats['no_face_files']:,}")
        print(f"   🔄 Duplicate groups: {quality_stats['duplicate_groups']:,}")
        print(f"   ⚠️  Cross-emotion contamination: {quality_stats['cross_emotion_duplicates']:,}")
        print(f"   🚨 Cross-split contamination: {quality_stats['cross_split_duplicates']:,}")
        
        if flagged_files:
            print(f"   🎯 Evidence clusters found: {len(flagged_files)}")
    
    return system

if __name__ == "__main__":
    system = run_sklearn_emotion_detection_system()
