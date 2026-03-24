"""
Healthcare Intelligence Layer
Clinical decision support with ethical constraints and bias awareness.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)


class ClinicalUrgency(str, Enum):
    """Clinical urgency levels for triage."""
    EMERGENT = "emergent"       # Immediate medical attention
    URGENT = "urgent"           # Within 24-48 hours
    SEMI_URGENT = "semi_urgent" # Within 1-2 weeks
    ROUTINE = "routine"         # Regular follow-up


@dataclass
class ClinicalInterpretation:
    """Clinical interpretation of model prediction."""
    risk_category: str
    urgency: ClinicalUrgency
    confidence_level: str
    key_risk_factors: List[str]
    protective_factors: List[str]
    clinical_considerations: List[str]
    recommended_actions: List[str]
    follow_up_timeframe: str
    specialist_referral: Optional[str]


class HealthcareIntelligence:
    """
    Healthcare-specific intelligence layer.
    
    Provides:
    - Clinical risk categorization
    - Threshold selection with clinical rationale
    - False negative risk handling
    - Ethical limitations and bias awareness
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Risk category definitions (probability thresholds)
        self.risk_categories = {
            "diabetes": {
                "low": (0, 0.2),
                "moderate": (0.2, 0.4),
                "high": (0.4, 0.6),
                "very_high": (0.6, 1.0)
            },
            "heart_disease": {
                "low": (0, 0.15),
                "moderate": (0.15, 0.35),
                "high": (0.35, 0.55),
                "very_high": (0.55, 1.0)
            },
            "stroke": {
                "low": (0, 0.1),
                "moderate": (0.1, 0.25),
                "high": (0.25, 0.45),
                "very_high": (0.45, 1.0)
            }
        }
    
    def get_risk_category(
        self,
        disease_type: str,
        probability: float
    ) -> Tuple[str, str]:
        """
        Get risk category and description.
        
        Returns:
            (category_name, description)
        """
        categories = self.risk_categories.get(disease_type, {
            "low": (0, 0.25),
            "moderate": (0.25, 0.5),
            "high": (0.5, 0.75),
            "very_high": (0.75, 1.0)
        })
        
        for category, (low, high) in categories.items():
            if low <= probability < high:
                descriptions = {
                    "low": "Risk is within normal population range",
                    "moderate": "Elevated risk - lifestyle modifications recommended",
                    "high": "Significantly elevated risk - medical consultation advised",
                    "very_high": "Very high risk - urgent medical attention recommended"
                }
                return category, descriptions[category]
        
        return "very_high", "Unable to categorize - please consult healthcare provider"
    
    def get_clinical_urgency(
        self,
        disease_type: str,
        probability: float,
        patient_factors: Optional[Dict[str, Any]] = None
    ) -> ClinicalUrgency:
        """
        Determine clinical urgency for triage.
        
        Considers:
        - Probability level
        - Disease severity (stroke > heart disease > diabetes for urgency)
        - Patient-specific factors (age, comorbidities)
        """
        # Base urgency from probability
        if probability >= 0.7:
            base_urgency = ClinicalUrgency.URGENT
        elif probability >= 0.5:
            base_urgency = ClinicalUrgency.SEMI_URGENT
        else:
            base_urgency = ClinicalUrgency.ROUTINE
        
        # Disease-specific escalation
        if disease_type == "stroke":
            # Stroke requires more aggressive triage
            if probability >= 0.5:
                base_urgency = ClinicalUrgency.URGENT
            elif probability >= 0.3:
                base_urgency = ClinicalUrgency.SEMI_URGENT
        
        # Patient factor escalation
        if patient_factors:
            age = patient_factors.get("age", 0)
            
            # Elderly patients need faster evaluation
            if age >= 65 and probability >= 0.3:
                if base_urgency == ClinicalUrgency.ROUTINE:
                    base_urgency = ClinicalUrgency.SEMI_URGENT
            
            # Existing comorbidities escalate urgency
            has_comorbidities = (
                patient_factors.get("hypertension", 0) == 1 or
                patient_factors.get("heart_disease", 0) == 1
            )
            if has_comorbidities and probability >= 0.4:
                if base_urgency == ClinicalUrgency.SEMI_URGENT:
                    base_urgency = ClinicalUrgency.URGENT
        
        return base_urgency
    
    def get_threshold_rationale(self, disease_type: str) -> Dict[str, Any]:
        """
        Get threshold selection rationale for clinical context.
        
        Explains WHY a particular threshold was chosen, which is
        critical for clinical acceptance and regulatory compliance.
        """
        threshold = self.settings.get_threshold(disease_type)
        fn_cost = self.settings.get_fn_cost(disease_type)
        
        rationales = {
            "diabetes": {
                "threshold": threshold,
                "rationale": (
                    "Threshold optimized for high sensitivity (≥85% recall) because: "
                    "1) Early diabetes detection enables lifestyle interventions that can "
                    "prevent disease progression. "
                    "2) The cost of a missed diagnosis (diabetic complications, neuropathy, "
                    "retinopathy) far exceeds the cost of additional screening. "
                    "3) Follow-up testing (HbA1c, OGTT) is non-invasive and relatively inexpensive."
                ),
                "fn_cost_ratio": fn_cost,
                "clinical_implications": [
                    "Higher false positive rate is acceptable - further testing will confirm",
                    "Lower threshold catches pre-diabetic patients for early intervention",
                    "Aligned with ADA guidelines for diabetes screening"
                ]
            },
            "heart_disease": {
                "threshold": threshold,
                "rationale": (
                    "Threshold optimized for high sensitivity because: "
                    "1) Missed cardiovascular disease can result in MI, stroke, or death. "
                    "2) Cardiology workup (stress test, echo) is diagnostic and therapeutic. "
                    "3) Risk stratification enables preventive statin/aspirin therapy."
                ),
                "fn_cost_ratio": fn_cost,
                "clinical_implications": [
                    "Prioritizes catching high-risk patients over minimizing false alarms",
                    "Downstream testing is standard of care for flagged patients",
                    "Consistent with ACC/AHA cardiovascular risk guidelines"
                ]
            },
            "stroke": {
                "threshold": threshold,
                "rationale": (
                    "Lowest threshold used because: "
                    "1) Stroke has catastrophic outcomes (death, permanent disability). "
                    "2) Time-sensitive intervention (tPA within 4.5 hours) is critical. "
                    "3) Prevention through anticoagulation/BP control is highly effective. "
                    "4) The irreversible nature of stroke damage justifies aggressive screening."
                ),
                "fn_cost_ratio": fn_cost,
                "clinical_implications": [
                    "Very high sensitivity prioritized over specificity",
                    "Aligns with stroke prevention as highest medical priority",
                    "Patients flagged should receive comprehensive vascular workup"
                ]
            }
        }
        
        return rationales.get(disease_type, {
            "threshold": threshold,
            "rationale": "Standard classification threshold",
            "fn_cost_ratio": fn_cost
        })
    
    def get_false_negative_guidance(
        self,
        disease_type: str,
        probability: float,
        prediction: int
    ) -> Optional[Dict[str, Any]]:
        """
        Provide guidance for borderline cases where false negative risk is elevated.
        
        Critical for cases near the threshold where clinician judgment should override.
        """
        threshold = self.settings.get_threshold(disease_type)
        
        # Borderline zone: within 0.15 of threshold
        borderline_margin = 0.15
        is_borderline = abs(probability - threshold) < borderline_margin
        
        if not is_borderline or prediction == 1:
            return None
        
        return {
            "warning": "BORDERLINE CASE - Review Recommended",
            "explanation": (
                f"This case falls in the borderline zone (probability {probability:.1%} "
                f"vs threshold {threshold:.1%}). Model predictions in this range have "
                "higher uncertainty. Clinical judgment should be prioritized."
            ),
            "recommendations": [
                "Consider patient-specific risk factors not captured by the model",
                "Review family history and social determinants of health",
                "Discuss risk factors directly with patient",
                "Consider additional diagnostic testing if clinical suspicion is high",
                "Schedule follow-up evaluation in 3-6 months"
            ],
            "clinical_override_suggested": probability > (threshold - borderline_margin / 2)
        }
    
    def get_interpretation(
        self,
        disease_type: str,
        probability: float,
        features: Dict[str, Any],
        risk_factors: List[Dict[str, Any]]
    ) -> ClinicalInterpretation:
        """
        Generate comprehensive clinical interpretation.
        """
        # Risk category
        category, category_desc = self.get_risk_category(disease_type, probability)
        
        # Urgency
        urgency = self.get_clinical_urgency(disease_type, probability, features)
        
        # Confidence level
        if 0.4 <= probability <= 0.6:
            confidence = "moderate"
        else:
            confidence = "high"
        
        # Separate risk and protective factors
        key_risks = [
            f["feature"] for f in risk_factors 
            if f.get("impact") == "increases"
        ][:5]
        
        protective = [
            f["feature"] for f in risk_factors
            if f.get("impact") == "decreases"
        ][:3]
        
        # Clinical considerations
        considerations = self._get_clinical_considerations(disease_type, features)
        
        # Recommended actions
        actions = self._get_recommended_actions(disease_type, category, features)
        
        # Follow-up timeframe
        follow_up = self._get_follow_up_timeframe(urgency)
        
        # Specialist referral
        referral = self._get_specialist_referral(disease_type, category)
        
        return ClinicalInterpretation(
            risk_category=category,
            urgency=urgency,
            confidence_level=confidence,
            key_risk_factors=key_risks,
            protective_factors=protective,
            clinical_considerations=considerations,
            recommended_actions=actions,
            follow_up_timeframe=follow_up,
            specialist_referral=referral
        )
    
    def _get_clinical_considerations(
        self,
        disease_type: str,
        features: Dict[str, Any]
    ) -> List[str]:
        """Get disease-specific clinical considerations."""
        considerations = []
        
        age = features.get("age", 0)
        
        if disease_type == "diabetes":
            if age < 40:
                considerations.append(
                    "Young age - consider Type 1 diabetes or MODY if positive"
                )
            if features.get("family_history", 0) == 1:
                considerations.append(
                    "Family history present - genetic predisposition likely"
                )
            if features.get("bmi", 0) > 35:
                considerations.append(
                    "Severe obesity - bariatric surgery may be indicated"
                )
        
        elif disease_type == "heart_disease":
            if age < 45 and features.get("gender") == "male":
                considerations.append(
                    "Young male with risk factors - investigate premature CAD"
                )
            if features.get("chest_pain_type", 0) >= 3:
                considerations.append(
                    "Typical anginal symptoms - high pre-test probability"
                )
        
        elif disease_type == "stroke":
            if features.get("hypertension", 0) == 1:
                considerations.append(
                    "Hypertension is the #1 modifiable risk factor - optimize BP control"
                )
            if features.get("heart_disease", 0) == 1:
                considerations.append(
                    "Cardiac source possible - consider anticoagulation evaluation"
                )
        
        return considerations
    
    def _get_recommended_actions(
        self,
        disease_type: str,
        risk_category: str,
        features: Dict[str, Any]
    ) -> List[str]:
        """Get recommended clinical actions based on risk level."""
        actions = []
        
        if risk_category in ["high", "very_high"]:
            if disease_type == "diabetes":
                actions.extend([
                    "Order HbA1c and fasting glucose",
                    "Refer to diabetes educator",
                    "Initiate lifestyle modification program"
                ])
            elif disease_type == "heart_disease":
                actions.extend([
                    "Order lipid panel and cardiac enzymes",
                    "Consider stress testing or cardiac catheterization",
                    "Initiate statin therapy per guidelines"
                ])
            elif disease_type == "stroke":
                actions.extend([
                    "Order carotid ultrasound",
                    "Consider echocardiogram for cardiac source",
                    "Optimize antihypertensive therapy",
                    "Consider antiplatelet or anticoagulation therapy"
                ])
        else:
            actions.extend([
                "Continue routine health maintenance",
                "Lifestyle counseling (diet, exercise, smoking cessation)",
                "Follow up in 6-12 months or as clinically indicated"
            ])
        
        return actions
    
    def _get_follow_up_timeframe(self, urgency: ClinicalUrgency) -> str:
        """Get follow-up timeframe based on urgency."""
        timeframes = {
            ClinicalUrgency.EMERGENT: "Immediate / Same day",
            ClinicalUrgency.URGENT: "Within 24-48 hours",
            ClinicalUrgency.SEMI_URGENT: "Within 1-2 weeks",
            ClinicalUrgency.ROUTINE: "Within 3-6 months"
        }
        return timeframes.get(urgency, "As clinically indicated")
    
    def _get_specialist_referral(
        self,
        disease_type: str,
        risk_category: str
    ) -> Optional[str]:
        """Get specialist referral recommendation."""
        if risk_category not in ["high", "very_high"]:
            return None
        
        referrals = {
            "diabetes": "Endocrinology",
            "heart_disease": "Cardiology",
            "stroke": "Neurology / Vascular Surgery"
        }
        return referrals.get(disease_type)


# Model limitations and ethical considerations
MODEL_LIMITATIONS = """
# Healthcare Prediction Model Limitations

## Intended Use
This model is designed as a SCREENING TOOL to assist healthcare professionals
in identifying patients who may benefit from additional evaluation. It is NOT
intended for:
- Definitive diagnosis
- Treatment decisions without clinical evaluation
- Emergency triage without human oversight
- Use as the sole basis for clinical action

## Known Limitations

### Data Limitations
1. **Training Data**: Models trained on synthetic/limited real-world data
2. **Population Representativeness**: May not generalize to all populations
3. **Temporal Validity**: Healthcare patterns change over time
4. **Missing Variables**: Many important clinical factors not captured

### Model Limitations
1. **Probability Calibration**: Despite calibration, probabilities are estimates
2. **Feature Interactions**: Complex interactions may not be fully captured
3. **Rare Outcomes**: Performance may degrade for rare conditions
4. **Out-of-Distribution**: Unusual cases may yield unreliable predictions

### Clinical Limitations
1. **False Negatives**: Some true positives will be missed
2. **False Positives**: Some healthy patients will be flagged
3. **Threshold Sensitivity**: Performance varies with threshold choice
4. **Context Blindness**: Cannot incorporate clinical gestalt

## Bias Considerations

### Potential Sources of Bias
1. **Selection Bias**: Training data may over/under-represent certain groups
2. **Measurement Bias**: Feature collection may vary by demographic
3. **Historical Bias**: Past healthcare disparities reflected in data
4. **Algorithmic Bias**: Model may perform differently across subgroups

### Mitigation Strategies
1. Regular bias auditing across demographic subgroups
2. Monitoring for disparate impact in predictions
3. Transparent reporting of subgroup performance
4. Human oversight for all clinical decisions

## Ethical Considerations

### Autonomy
- Patients should be informed about AI-assisted screening
- Shared decision-making should be prioritized
- Right to refuse AI-based evaluation should be respected

### Beneficence
- Model should improve health outcomes on average
- Regular validation against clinical endpoints required
- Continuous improvement based on feedback

### Non-maleficence
- False negatives must be minimized for serious conditions
- Overtreatment from false positives must be considered
- Psychological impact of risk scores should be managed

### Justice
- Equal access to AI-assisted screening
- Monitoring for disparate impact across populations
- Transparent about limitations in underrepresented groups

## Regulatory Compliance
This tool should be used in accordance with:
- FDA guidance on clinical decision support software
- HIPAA requirements for protected health information
- Institutional review board approval where applicable
- Local medical device regulations
"""

DISCLAIMERS = {
    "standard": (
        "⚠️ IMPORTANT DISCLAIMER: This prediction is generated by a machine learning "
        "model and is intended for informational and educational purposes only. It does "
        "NOT constitute medical advice, diagnosis, or treatment recommendation. The "
        "predictions should be used as one of many inputs in clinical decision-making, "
        "not as a substitute for professional medical judgment. Always consult with "
        "qualified healthcare professionals for medical decisions. Do not delay seeking "
        "medical attention or disregard professional medical advice based on this prediction."
    ),
    "clinical": (
        "CLINICAL DECISION SUPPORT NOTICE: This tool is designed to assist clinical "
        "decision-making, not replace it. Model predictions should be interpreted in "
        "the context of complete clinical presentation, patient history, and professional "
        "judgment. The model has known limitations including potential biases and "
        "reduced accuracy in certain populations. Clinicians retain full responsibility "
        "for patient care decisions."
    ),
    "research": (
        "RESEARCH USE ONLY: This model has not been validated for clinical use and "
        "should not be used for patient care decisions. It is provided for research "
        "and educational purposes only."
    )
}
