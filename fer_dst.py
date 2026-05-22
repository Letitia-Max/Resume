import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import FER, fall back to mock if not available
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    print("⚠️ FER library not found. Using mock FER for demonstration.")
    FER_AVAILABLE = False
    
    class MockFER:
        def __init__(self, mtcnn=True):
            pass
        
        def detect_emotions(self, img):
            # Mock emotion detection for demonstration
            emotions = {
                'angry': np.random.random() * 0.2,
                'disgust': np.random.random() * 0.1,
                'fear': np.random.random() * 0.2,
                'happy': np.random.random() * 0.4,
                'sad': np.random.random() * 0.3,
                'surprise': np.random.random() * 0.3,
                'neutral': np.random.random() * 0.5
            }
            # Normalize to sum to 1
            total = sum(emotions.values())
            emotions = {k: v/total for k, v in emotions.items()}
            
            return [{
                'box': [50, 50, 100, 100],
                'emotions': emotions
            }]

class FERDSTConfidenceAnalyzer:
    """
    Enhanced FER + DST + Confidence Intervals Demographic Bias Analyzer
    
    Pipeline: FER Detection → DST Uncertainty Analysis → Statistical Validation → Deployment Decision
    
    Enhanced features beyond baseline:
    - Null hypothesis testing with p-values
    - Enhanced confidence intervals with statistical rigor
    - Per-emotion statistical significance testing
    - Effect size calculations
    """

    def __init__(self, dataset_path, uncertainty_threshold=0.3, confidence_level=0.95):
        self.dataset_path = Path(dataset_path)
        self.results = defaultdict(list)
        self.demographic_stats = defaultdict(dict)
        
        # FER Integration
        if FER_AVAILABLE:
            self.fer_detector = FER(mtcnn=True)
        else:
            self.fer_detector = MockFER()
            
        # DST Parameters
        self.uncertainty_threshold = uncertainty_threshold
        
        # Enhanced Confidence Interval Parameters
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # For 95% CI, alpha = 0.05
        
        # Performance tracking for confidence intervals
        self.prediction_history = []
        self.accuracy_history = []

    def estimate_skin_tone(self, image_path, method='hsv_analysis'):
        """
        Estimate skin tone from facial image using multiple methods
        Returns a score where lower values suggest darker skin tones
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # Convert to different color spaces for analysis
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Method 1: Average luminance (simple but effective for grayscale)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_luminance = np.mean(gray)

            # Method 2: Face region detection (more accurate)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Use the largest detected face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

                # Extract face region for more accurate skin tone analysis
                face_region = img_rgb[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]

                # Calculate multiple skin tone indicators
                face_luminance = np.mean(face_gray)

                # HSV analysis - look at Value channel
                face_hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
                avg_value = np.mean(face_hsv[:,:,2])

                # LAB analysis - L channel represents lightness
                face_lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
                avg_lightness = np.mean(face_lab[:,:,0])

                return {
                    'luminance': face_luminance,
                    'hsv_value': avg_value,
                    'lab_lightness': avg_lightness,
                    'composite_score': (face_luminance + avg_value + avg_lightness/2.55) / 3,
                    'face_detected': True
                }
            else:
                # Fallback to whole image analysis
                avg_value = np.mean(img_hsv[:,:,2])
                avg_lightness = np.mean(img_lab[:,:,0])

                return {
                    'luminance': avg_luminance,
                    'hsv_value': avg_value,
                    'lab_lightness': avg_lightness,
                    'composite_score': (avg_luminance + avg_value + avg_lightness/2.55) / 3,
                    'face_detected': False
                }

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def categorize_skin_tone(self, skin_metrics):
        """
        Categorize skin tone based on composite metrics
        """
        if skin_metrics is None:
            return 'unknown'

        composite = skin_metrics['composite_score']

        
        # Lower scores indicate darker skin tones
        if composite < 80:
            return 'darker'
        elif composite < 120:
            return 'medium'
        else:
            return 'lighter'

    def fer_emotion_detection(self, image_path):
        """
        FER-based emotion detection with error handling
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # FER detection
            fer_results = self.fer_detector.detect_emotions(img_rgb)
            
            if not fer_results:
                return {
                    'emotions': {},
                    'dominant_emotion': 'unknown',
                    'confidence': 0.0,
                    'fer_detected': False
                }
            
            # Take the first (largest) face
            face_result = fer_results[0]
            emotions = face_result['emotions']
            
            # Map to our 4-class system
            emotion_mapping = {
                'happy': emotions.get('happy', 0),
                'sad': emotions.get('sad', 0),
                'neutral': emotions.get('neutral', 0),
                'surprise': emotions.get('surprise', 0)
            }
            
            # Find dominant emotion from our 4 classes
            dominant_emotion = max(emotion_mapping, key=emotion_mapping.get)
            confidence = emotion_mapping[dominant_emotion]
            
            return {
                'emotions': emotion_mapping,
                'full_emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'fer_detected': True
            }
            
        except Exception as e:
            print(f"FER error for {image_path}: {e}")
            return {
                'emotions': {},
                'dominant_emotion': 'error',
                'confidence': 0.0,
                'fer_detected': False
            }

    def dst_uncertainty_analysis(self, fer_result, skin_metrics, image_path):
        """
        DST-based uncertainty analysis combining FER and demographic data
        """
        evidence_sources = {}
        
        # Evidence 1: FER Reliability
        fer_confidence = fer_result.get('confidence', 0.0)
        fer_detected = fer_result.get('fer_detected', False)
        
        evidence_sources['fer_reliability'] = {
            'confidence': fer_confidence,
            'detected': fer_detected,
            'reliability': fer_confidence if fer_detected else 0.1
        }
        
        # Evidence 2: Face Detection Quality
        face_detected = skin_metrics.get('face_detected', False) if skin_metrics else False
        face_quality = skin_metrics.get('composite_score', 0.0) if skin_metrics else 0.0
        
        evidence_sources['face_quality'] = {
            'detected': face_detected,
            'quality_score': face_quality,
            'reliability': (face_quality / 100.0) if face_detected else 0.2
        }
        
        # Evidence 3: Demographic Bias Risk
        skin_category = self.categorize_skin_tone(skin_metrics) if skin_metrics else 'unknown'
        emotion = fer_result.get('dominant_emotion', 'unknown')
        
        # Calculate bias risk based on known patterns
        bias_risk = self._calculate_bias_risk(skin_category, emotion)
        
        evidence_sources['bias_risk'] = {
            'skin_category': skin_category,
            'emotion': emotion,
            'bias_risk_score': bias_risk,
            'reliability': 1.0 - bias_risk  # Higher bias = lower reliability
        }
        
        # DST Evidence Combination
        combined_confidence = self._dst_combine_evidence(evidence_sources)
        
        # Generate explanation
        explanation = self._generate_dst_explanation(
            fer_result, combined_confidence, evidence_sources
        )
        
        return {
            'combined_confidence': combined_confidence['final_confidence'],
            'uncertainty_score': combined_confidence['uncertainty'],
            'evidence_sources': evidence_sources,
            'explanation': explanation,
            'needs_review': combined_confidence['uncertainty'] > self.uncertainty_threshold,
            'bias_risk': bias_risk
        }

    def _calculate_bias_risk(self, skin_category, emotion):
        """
        Calculate bias risk based on demographic-emotion combinations
        """
        # Known bias patterns from analysis
        high_risk_combinations = {
            ('darker', 'sad'): 0.8,      # High bias risk
            ('darker', 'angry'): 0.9,    # Very high bias risk  
            ('lighter', 'happy'): 0.3,   # Low bias risk
            ('darker', 'surprise'): 0.7, # Underrepresentation risk
        }
        
        # Default moderate risk for unknown combinations
        return high_risk_combinations.get((skin_category, emotion), 0.5)

    def _dst_combine_evidence(self, evidence_sources):
        """
        Simplified DST evidence combination with weighted fusion
        """
        weights = {
            'fer_reliability': 0.4,
            'face_quality': 0.35,
            'bias_risk': 0.25
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

    def _generate_dst_explanation(self, fer_result, combined_confidence, evidence_sources):
        """
        Generate human-readable DST explanation
        """
        emotion = fer_result.get('dominant_emotion', 'unknown')
        confidence = combined_confidence['final_confidence']
        uncertainty = combined_confidence['uncertainty']
        
        if uncertainty > 0.7:
            return f"High uncertainty for '{emotion}' prediction. Multiple evidence sources indicate reliability issues."
        elif uncertainty > 0.4:
            return f"Moderate uncertainty for '{emotion}' prediction. DST confidence: {confidence:.2f}."
        else:
            return f"Low uncertainty for '{emotion}' prediction. DST validates reliability: {confidence:.2f}."

    def calculate_confidence_intervals(self, predictions_df):
        """
        Enhanced confidence intervals with null hypothesis testing for statistical rigor
        """
        if len(predictions_df) == 0:
            return None
            
        # Filter for DST-approved predictions only
        approved_predictions = predictions_df[predictions_df['dst_approved'] == True]
        
        if len(approved_predictions) == 0:
            return {
                'message': 'No DST-approved predictions for confidence interval calculation',
                'n_approved': 0,
                'total_predictions': len(predictions_df)
            }
        
        # Calculate performance metrics on approved predictions
        if 'true_emotion' in approved_predictions.columns:
            # Calculate accuracy for approved predictions
            correct_predictions = (approved_predictions['predicted_emotion'] == 
                                 approved_predictions['true_emotion']).sum()
            n_approved = len(approved_predictions)
            accuracy = correct_predictions / n_approved
            
            # Binomial confidence interval for accuracy
            ci_lower, ci_upper = self._binomial_confidence_interval(
                correct_predictions, n_approved, self.confidence_level
            )
            
            # ENHANCED: NULL HYPOTHESIS TESTING
            # H0: Model performs no better than random chance (25% for 4-class problem)
            # H1: Model performs better than random chance
            null_accuracy = 0.25  # Random chance for 4-class classification
            
            # Binomial test for null hypothesis
            p_value = self._binomial_test(correct_predictions, n_approved, null_accuracy)
            
            # Statistical significance test
            alpha = 1 - self.confidence_level  # 0.05 for 95% CI
            reject_null = p_value < alpha
            
            # Effect size (difference from null hypothesis)
            effect_size = accuracy - null_accuracy
            
            # ENHANCED: Per-emotion confidence intervals and null hypothesis tests
            emotion_cis = {}
            for emotion in ['happy', 'sad', 'neutral', 'surprise']:
                emotion_subset = approved_predictions[
                    approved_predictions['true_emotion'] == emotion
                ]
                if len(emotion_subset) > 0:
                    emotion_correct = (emotion_subset['predicted_emotion'] == emotion).sum()
                    emotion_n = len(emotion_subset)
                    emotion_acc = emotion_correct / emotion_n
                    
                    emotion_ci_lower, emotion_ci_upper = self._binomial_confidence_interval(
                        emotion_correct, emotion_n, self.confidence_level
                    )
                    
                    # Per-emotion null hypothesis test (25% random chance)
                    emotion_p_value = self._binomial_test(emotion_correct, emotion_n, null_accuracy)
                    emotion_reject_null = emotion_p_value < alpha
                    
                    emotion_cis[emotion] = {
                        'accuracy': emotion_acc,
                        'ci_lower': emotion_ci_lower,
                        'ci_upper': emotion_ci_upper,
                        'n_samples': emotion_n,
                        'p_value': emotion_p_value,
                        'reject_null_hypothesis': emotion_reject_null,
                        'effect_size': emotion_acc - null_accuracy
                    }
            
            return {
                'overall_accuracy': accuracy,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_approved': n_approved,
                'total_predictions': len(predictions_df),
                'approval_rate': n_approved / len(predictions_df),
                'confidence_level': self.confidence_level,
                'emotion_specific_cis': emotion_cis,
                # ENHANCED: NULL HYPOTHESIS TESTING RESULTS
                'null_hypothesis_test': {
                    'null_accuracy': null_accuracy,
                    'observed_accuracy': accuracy,
                    'p_value': p_value,
                    'reject_null_hypothesis': reject_null,
                    'effect_size': effect_size,
                    'interpretation': self._interpret_null_hypothesis_test(accuracy, p_value, reject_null)
                }
            }
        else:
            return {
                'message': 'True labels not available for confidence interval calculation',
                'n_approved': len(approved_predictions),
                'total_predictions': len(predictions_df)
            }

    def _binomial_confidence_interval(self, successes, trials, confidence_level):
        """
        Calculate binomial confidence interval using Wilson score method
        """
        if trials == 0:
            return 0.0, 0.0
            
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        p = successes / trials
        
        # Wilson score interval (more accurate for small samples)
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return max(0, centre - half_width), min(1, centre + half_width)
    
    def _binomial_test(self, successes, trials, null_probability):
        """
        Enhanced binomial test for null hypothesis testing
        """
        if trials == 0:
            return 1.0
        
        # Two-tailed binomial test
        observed_prob = successes / trials
        
        # Calculate p-value using scipy's binomial test
        # For two-tailed test: probability of observing this extreme or more extreme result
        from scipy.stats import binom
        
        if observed_prob >= null_probability:
            # Upper tail: P(X >= observed)
            p_value = 1 - binom.cdf(successes - 1, trials, null_probability)
        else:
            # Lower tail: P(X <= observed)  
            p_value = binom.cdf(successes, trials, null_probability)
        
        # Two-tailed test: multiply by 2, but cap at 1.0
        p_value = min(2 * p_value, 1.0)
        
        return p_value
    
    def _interpret_null_hypothesis_test(self, accuracy, p_value, reject_null):
        """
        Generate interpretation of null hypothesis test results
        """
        if reject_null:
            if accuracy > 0.25:
                return f"REJECT NULL: Model performs significantly better than random chance (p={p_value:.4f})"
            else:
                return f"REJECT NULL: Model performs significantly worse than random chance (p={p_value:.4f})"
        else:
            return f"FAIL TO REJECT NULL: No significant difference from random chance detected (p={p_value:.4f})"

    def analyze_image_with_full_pipeline(self, image_path, true_emotion=None):
        """
        Complete pipeline: FER → DST → Statistical Validation
        """
        # Step 1: Demographic Analysis
        skin_metrics = self.estimate_skin_tone(image_path)
        skin_category = self.categorize_skin_tone(skin_metrics)
        
        # Step 2: FER Emotion Detection
        fer_result = self.fer_emotion_detection(image_path)
        
        # Step 3: DST Uncertainty Analysis
        dst_analysis = self.dst_uncertainty_analysis(fer_result, skin_metrics, image_path)
        
        # Step 4: Deployment Decision
        dst_approved = not dst_analysis['needs_review']
        
        # Compile results
        result = {
            'image_path': str(image_path),
            'skin_category': skin_category,
            'skin_metrics': skin_metrics,
            'fer_result': fer_result,
            'dst_analysis': dst_analysis,
            'predicted_emotion': fer_result.get('dominant_emotion', 'unknown'),
            'dst_approved': dst_approved,
            'true_emotion': true_emotion,
            'timestamp': pd.Timestamp.now()
        }
        
        return result

    def analyze_dataset_with_pipeline(self):
        """
        Analyze entire dataset with enhanced FER + DST + Statistical Validation pipeline
        """
        print("🎭 Starting Enhanced FER + DST + Statistical Validation Pipeline")
        print("="*70)

        emotions = ['happy', 'sad', 'neutral', 'surprise']
        splits = ['train', 'validation', 'test']
        
        all_results = []
        total_processed = 0

        for split in splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue

            for emotion in emotions:
                emotion_path = split_path / emotion
                if not emotion_path.exists():
                    continue

                image_files = list(emotion_path.glob('*.jpg')) + list(emotion_path.glob('*.png'))
                
                print(f"📊 Processing {split}/{emotion}: {len(image_files)} images")

                for img_path in image_files:
                    # Run full pipeline
                    result = self.analyze_image_with_full_pipeline(img_path, true_emotion=emotion)
                    result['split'] = split
                    all_results.append(result)
                    
                    total_processed += 1
                    if total_processed % 50 == 0:
                        print(f"   ✅ Processed {total_processed} images...")

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Step 5: Enhanced Statistical Analysis (after DST filtering)
        ci_results = self.calculate_confidence_intervals(results_df)
        
        print(f"\n✅ Enhanced pipeline analysis complete! Processed {total_processed} images")
        
        return results_df, ci_results

    def generate_pipeline_report(self, results_df, ci_results):
        """
        Generate comprehensive enhanced pipeline analysis report
        """
        print("\n" + "="*80)
        print("🎭 ENHANCED FER + DST + STATISTICAL VALIDATION PIPELINE REPORT")
        print("="*80)
        
        total_images = len(results_df)
        
        # FER Performance
        fer_success = results_df['fer_result'].apply(lambda x: x.get('fer_detected', False)).sum()
        print(f"\n🤖 FER EMOTION DETECTION:")
        print(f"   Total images: {total_images:,}")
        print(f"   FER successful: {fer_success:,} ({fer_success/total_images*100:.1f}%)")
        
        # DST Analysis
        dst_approved = results_df['dst_approved'].sum()
        high_uncertainty = results_df['dst_analysis'].apply(lambda x: x.get('uncertainty_score', 0) > 0.7).sum()
        
        print(f"\n🧠 DST UNCERTAINTY ANALYSIS:")
        print(f"   DST approved for deployment: {dst_approved:,} ({dst_approved/total_images*100:.1f}%)")
        print(f"   High uncertainty cases: {high_uncertainty:,} ({high_uncertainty/total_images*100:.1f}%)")
        print(f"   Human review required: {total_images - dst_approved:,} ({(total_images - dst_approved)/total_images*100:.1f}%)")
        
        # Enhanced Confidence Intervals with NULL HYPOTHESIS TESTING
        if ci_results and 'overall_accuracy' in ci_results:
            print(f"\n📊 ENHANCED CONFIDENCE INTERVALS (DST-Approved Predictions Only):")
            print(f"   Overall Accuracy: {ci_results['overall_accuracy']:.3f}")
            print(f"   95% CI: [{ci_results['ci_lower']:.3f}, {ci_results['ci_upper']:.3f}]")
            print(f"   Samples in CI calculation: {ci_results['n_approved']:,}")
            print(f"   DST Approval Rate: {ci_results['approval_rate']:.1%}")
            
            # ENHANCED: NULL HYPOTHESIS TEST RESULTS
            if 'null_hypothesis_test' in ci_results:
                null_test = ci_results['null_hypothesis_test']
                print(f"\n🧪 STATISTICAL SIGNIFICANCE TESTING:")
                print(f"   H0: Model accuracy = {null_test['null_accuracy']:.1%} (random chance)")
                print(f"   H1: Model accuracy ≠ {null_test['null_accuracy']:.1%}")
                print(f"   Observed accuracy: {null_test['observed_accuracy']:.3f}")
                print(f"   P-value: {null_test['p_value']:.4f}")
                print(f"   Statistical Result: {null_test['interpretation']}")
                print(f"   Effect size: {null_test['effect_size']:+.3f}")
            
            if 'emotion_specific_cis' in ci_results:
                print(f"\n📋 PER-EMOTION STATISTICAL ANALYSIS:")
                for emotion, stats in ci_results['emotion_specific_cis'].items():
                    print(f"   {emotion.capitalize()}:")
                    print(f"     Accuracy: {stats['accuracy']:.3f}")
                    print(f"     95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
                    print(f"     Samples: {stats['n_samples']}")
                    if 'p_value' in stats:
                        print(f"     P-value vs null: {stats['p_value']:.4f}")
                        null_result = "REJECT H0" if stats['reject_null_hypothesis'] else "FAIL TO REJECT H0"
                        print(f"     Statistical significance: {null_result}")
                        print(f"     Effect size: {stats['effect_size']:+.3f}")
        
        # Bias Analysis
        print(f"\n⚠️  BIAS RISK ANALYSIS:")
        high_bias_risk = results_df['dst_analysis'].apply(lambda x: x.get('bias_risk', 0) > 0.7).sum()
        print(f"   High bias risk predictions: {high_bias_risk:,} ({high_bias_risk/total_images*100:.1f}%)")
        
        # Enhanced Academic Summary
        print(f"\n🎓 ACADEMIC VALIDATION SUMMARY:")
        if ci_results and 'null_hypothesis_test' in ci_results:
            null_test = ci_results['null_hypothesis_test']
            print(f"   Statistical significance: {'YES' if null_test['reject_null_hypothesis'] else 'NO'}")
            print(f"   Academic rigor: Enhanced with null hypothesis testing")
            print(f"   Peer review ready: {'YES' if null_test['p_value'] < 0.05 else 'NEEDS REVIEW'}")
        
        return results_df

    def print_summary(self):
        """
        Enhanced summary with statistical validation information
        """
        print("\n" + "="*60)
        print("ENHANCED FER + DST + STATISTICAL VALIDATION SUMMARY")
        print("="*60)

        total_images = len(self.df) if hasattr(self, 'df') else 0
        print(f"Total images analyzed: {total_images}")

        if hasattr(self, 'df') and total_images > 0:
            # Skin tone distribution
            skin_dist = self.df['skin_category'].value_counts()
            print("\nSkin tone distribution:")
            for category, count in skin_dist.items():
                pct = (count / total_images) * 100
                print(f"  {category}: {count} ({pct:.1f}%)")

            # Check for potential bias patterns
            print("\n🚨 ENHANCED BIAS INDICATORS:")

            # Check if darker skin tones are overrepresented in negative emotions
            darker_data = self.df[self.df['skin_category'] == 'darker']
            if len(darker_data) > 0:
                darker_emotions = darker_data['emotion_label'].value_counts(normalize=True)
                negative_emotions = darker_emotions.get('sad', 0)
                print(f"Darker skin - sad emotions: {negative_emotions:.1%}")

        print("="*60)

# Usage example and main execution
def main():
    """
    Main execution function with enhanced statistical validation pipeline
    """
    # Set your dataset path here
    dataset_path = r"C:\projects\AI\Facial_emotion_images"

    # Initialize enhanced analyzer
    analyzer = FERDSTConfidenceAnalyzer(dataset_path, uncertainty_threshold=0.3, confidence_level=0.95)

    # Run enhanced pipeline analysis
    print("🚀 Starting Enhanced FER + DST + Statistical Validation Analysis...")
    results_df, ci_results = analyzer.analyze_dataset_with_pipeline()

    # Generate comprehensive report
    analyzer.generate_pipeline_report(results_df, ci_results)

    # Save results
    results_df.to_csv('fer_dst_ci_analysis.csv', index=False)
    print(f"\n💾 Enhanced pipeline results saved to: fer_dst_ci_analysis.csv")

    # === Automatic DST split for training files ===
    dst_true_csv = 'fer_dst_true.csv'
    dst_false_csv = 'fer_dst_false.csv'
    df_true = results_df[results_df['dst_approved'] == True]
    df_true.to_csv(dst_true_csv, index=False)
    print(f"💾 Saved {len(df_true)} DST=True samples to {dst_true_csv}")
    df_false = results_df[results_df['dst_approved'] == False]
    df_false.to_csv(dst_false_csv, index=False)
    print(f"💾 Saved {len(df_false)} DST=False samples to {dst_false_csv}")
    print(f"Summary: {len(df_true)} DST=True | {len(df_false)} DST=False | {len(results_df)} total")

    # === Enhanced Visualizations and Reports ===
    def create_fer_dst_summary_visualizations(results_df):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # 1. Confusion Matrix (FER vs Ground Truth)
        from sklearn.metrics import confusion_matrix
        emotions = ['happy', 'sad', 'neutral', 'surprise']
        y_true = results_df['true_emotion']
        y_pred = results_df['predicted_emotion']
        cm = confusion_matrix(y_true, y_pred, labels=emotions)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FER + DST Bias-Aware Emotion Analysis', fontsize=16, fontweight='bold')

        # Confusion Matrix
        ax1 = axes[0, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions, ax=ax1)
        acc = (y_true == y_pred).mean()
        ax1.set_title(f'FER vs Ground Truth\nAccuracy: {acc:.2%}')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # DST Uncertainty Distribution
        ax2 = axes[0, 1]
        uncertainty = results_df['dst_analysis'].apply(lambda x: x.get('uncertainty_score', 0))
        ax2.hist(uncertainty, bins=20, color='orange', alpha=0.7)
        ax2.set_title('DST Uncertainty Score Distribution')
        ax2.set_xlabel('Uncertainty Score')
        ax2.set_ylabel('Count')

        # Bias Detection Heatmap (skin tone vs emotion)
        ax3 = axes[1, 0]
        bias_pivot = pd.pivot_table(results_df, index='skin_category', columns='predicted_emotion', values='image_path', aggfunc='count', fill_value=0)
        sns.heatmap(bias_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Predicted Emotion by Skin Tone')
        ax3.set_xlabel('Predicted Emotion')
        ax3.set_ylabel('Skin Tone')

        # FER Accuracy by Skin Tone
        ax4 = axes[1, 1]
        skin_tones = results_df['skin_category'].unique()
        acc_by_skin = []
        for skin in skin_tones:
            mask = results_df['skin_category'] == skin
            acc_by_skin.append((mask & (y_true == y_pred)).sum() / mask.sum() if mask.sum() > 0 else 0)
        ax4.bar(skin_tones, acc_by_skin, color='teal', alpha=0.7)
        ax4.set_title('FER Accuracy by Skin Tone')
        ax4.set_xlabel('Skin Tone')
        ax4.set_ylabel('Accuracy')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('fer_dst_summary_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Summary visualization saved: fer_dst_summary_results.png")

    def write_fer_dst_summary_metrics(results_df, ci_results):
        with open('fer_dst_summary_metrics.txt', 'w', encoding='utf-8') as f:
            total = len(results_df)
            acc = (results_df['true_emotion'] == results_df['predicted_emotion']).mean()
            dst_approved = results_df['dst_approved'].sum()
            high_uncertainty = results_df['dst_analysis'].apply(lambda x: x.get('uncertainty_score', 0) > 0.7).sum()
            f.write("FER + DST Bias-Aware Emotion Analysis - Summary Metrics\n")
            f.write("="*60 + "\n")
            f.write(f"Total images analyzed: {total}\n")
            f.write(f"FER Overall Accuracy: {acc:.3f} ({acc*100:.1f}%)\n")
            f.write(f"DST Approved for Deployment: {dst_approved} ({dst_approved/total*100:.1f}%)\n")
            f.write(f"High Uncertainty Cases: {high_uncertainty} ({high_uncertainty/total*100:.1f}%)\n")
            if ci_results and 'overall_accuracy' in ci_results:
                f.write(f"DST-Approved Accuracy: {ci_results['overall_accuracy']:.3f}\n")
                f.write(f"95% CI: [{ci_results['ci_lower']:.3f}, {ci_results['ci_upper']:.3f}]\n")
                f.write(f"Null Hypothesis p-value: {ci_results['null_hypothesis_test']['p_value']:.4f}\n")
                f.write(f"Statistical Result: {ci_results['null_hypothesis_test']['interpretation']}\n")
            f.write("="*60 + "\n")
        print("📝 Summary metrics saved: fer_dst_summary_metrics.txt")

    def write_fer_dst_bias_analysis_report(results_df):
        with open('fer_dst_bias_analysis_report.txt', 'w', encoding='utf-8') as f:
            total = len(results_df)
            f.write("FER + DST Bias-Aware Emotion Analysis - Bias Report\n")
            f.write("="*60 + "\n")
            # Skin tone distribution
            skin_dist = results_df['skin_category'].value_counts()
            f.write("Skin Tone Distribution:\n")
            for cat, count in skin_dist.items():
                f.write(f"  {cat}: {count} ({count/total*100:.1f}%)\n")
            # Bias risk
            high_bias = results_df['dst_analysis'].apply(lambda x: x.get('bias_risk', 0) > 0.7).sum()
            f.write(f"\nHigh Bias Risk Predictions: {high_bias} ({high_bias/total*100:.1f}%)\n")
            # Accuracy by skin tone
            f.write("\nFER Accuracy by Skin Tone:\n")
            for skin in skin_dist.index:
                mask = results_df['skin_category'] == skin
                acc = (results_df[mask]['true_emotion'] == results_df[mask]['predicted_emotion']).mean()
                f.write(f"  {skin}: {acc:.3f} ({acc*100:.1f}%)\n")
            # Most common mismatches
            mismatches = results_df[results_df['true_emotion'] != results_df['predicted_emotion']]
            if not mismatches.empty:
                f.write("\nMost Common Mismatches (Top 5):\n")
                mismatch_counts = mismatches.groupby(['skin_category', 'true_emotion', 'predicted_emotion']).size().sort_values(ascending=False).head(5)
                for idx, count in mismatch_counts.items():
                    f.write(f"  {idx}: {count}\n")
            f.write("="*60 + "\n")
        print("📝 Bias analysis report saved: fer_dst_bias_analysis_report.txt")

    # === Run new outputs ===
    create_fer_dst_summary_visualizations(results_df)
    write_fer_dst_summary_metrics(results_df, ci_results)
    write_fer_dst_bias_analysis_report(results_df)

    return analyzer, results_df, ci_results

if __name__ == "__main__":
    main()
