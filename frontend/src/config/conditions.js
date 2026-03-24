// Condition configurations
export const CONDITIONS = {
  diabetes: {
    id: 'diabetes',
    name: 'Diabetes',
    subtitle: 'Type 2 Diabetes Risk',
    category: 'Metabolic Health',
    icon: '🩸',
    color: 'blue',
    gradient: 'from-blue-500 to-blue-600',
    lightBg: 'bg-blue-50',
    description: 'Assess your risk factors for developing Type 2 diabetes based on lifestyle and health indicators.',
    facts: [
      'Over 37 million Americans have diabetes',
      'Early detection can prevent serious complications',
      'Lifestyle changes can reduce risk by up to 58%'
    ]
  },
  heart_disease: {
    id: 'heart_disease',
    name: 'Heart Disease',
    subtitle: 'Cardiovascular Risk',
    category: 'Cardiovascular Health',
    icon: '❤️',
    color: 'red',
    gradient: 'from-rose-500 to-red-600',
    lightBg: 'bg-rose-50',
    description: 'Evaluate your cardiovascular health risk based on key indicators and lifestyle factors.',
    facts: [
      'Heart disease is the leading cause of death globally',
      '80% of heart disease is preventable',
      'Regular screening saves lives'
    ]
  },
  stroke: {
    id: 'stroke',
    name: 'Stroke',
    subtitle: 'Cerebrovascular Risk',
    category: 'Neurological Health',
    icon: '🧠',
    color: 'purple',
    gradient: 'from-purple-500 to-purple-600',
    lightBg: 'bg-purple-50',
    description: 'Understand your stroke risk factors and learn how to protect your brain health.',
    facts: [
      'Stroke is a leading cause of disability',
      'Up to 80% of strokes are preventable',
      'Fast action is critical during a stroke'
    ]
  }
}

// Form field configurations for each condition
export const FORM_CONFIGS = {
  diabetes: {
    steps: [
      {
        title: 'Basic Information',
        description: 'Let\'s start with some basic details about you',
        fields: [
          { name: 'age', label: 'Age', type: 'slider', min: 18, max: 100, unit: 'years' },
          { name: 'gender', label: 'Gender', type: 'toggle', options: ['Male', 'Female'] },
          { name: 'bmi', label: 'Body Mass Index (BMI)', type: 'slider', min: 15, max: 50, step: 0.1, unit: 'kg/m²', hint: 'Weight(kg) / Height(m)²' },
        ]
      },
      {
        title: 'Health Metrics',
        description: 'Share your recent health measurements',
        fields: [
          { name: 'blood_pressure', label: 'Blood Pressure (Systolic)', type: 'slider', min: 80, max: 200, unit: 'mmHg' },
          { name: 'glucose', label: 'Fasting Blood Glucose', type: 'slider', min: 50, max: 300, unit: 'mg/dL' },
          { name: 'insulin', label: 'Insulin Level', type: 'slider', min: 0, max: 300, unit: 'μU/mL' },
        ]
      },
      {
        title: 'Lifestyle Factors',
        description: 'Tell us about your daily habits',
        fields: [
          { name: 'physical_activity', label: 'Physical Activity (days per week)', type: 'slider', min: 0, max: 7, unit: 'days' },
          { name: 'sleep_hours', label: 'Average Sleep', type: 'slider', min: 3, max: 12, unit: 'hours' },
          { name: 'stress_level', label: 'Stress Level', type: 'slider', min: 1, max: 10, unit: '/10' },
          { name: 'family_history', label: 'Family History of Diabetes', type: 'toggle', options: ['No', 'Yes'] },
          { name: 'smoking', label: 'Smoking Status', type: 'toggle', options: ['No', 'Yes'] },
          { name: 'alcohol', label: 'Regular Alcohol Consumption', type: 'toggle', options: ['No', 'Yes'] },
        ]
      }
    ]
  },
  heart_disease: {
    steps: [
      {
        title: 'Basic Information',
        description: 'Let\'s start with your basic health profile',
        fields: [
          { name: 'age', label: 'Age', type: 'slider', min: 18, max: 100, unit: 'years' },
          { name: 'gender', label: 'Gender', type: 'toggle', options: ['Male', 'Female'] },
          { name: 'bmi', label: 'Body Mass Index (BMI)', type: 'slider', min: 15, max: 50, step: 0.1, unit: 'kg/m²' },
        ]
      },
      {
        title: 'Cardiovascular Metrics',
        description: 'Your heart health indicators',
        fields: [
          { name: 'resting_bp', label: 'Resting Blood Pressure', type: 'slider', min: 80, max: 200, unit: 'mmHg' },
          { name: 'cholesterol', label: 'Total Cholesterol', type: 'slider', min: 100, max: 400, unit: 'mg/dL' },
          { name: 'max_heart_rate', label: 'Maximum Heart Rate', type: 'slider', min: 60, max: 220, unit: 'bpm' },
          { name: 'oldpeak', label: 'ST Depression (Exercise)', type: 'slider', min: 0, max: 6, step: 0.1, unit: 'mm' },
        ]
      },
      {
        title: 'Medical History',
        description: 'Your health background',
        fields: [
          { name: 'chest_pain_type', label: 'Chest Pain Type', type: 'select', options: [
            { value: 0, label: 'No chest pain' },
            { value: 1, label: 'Typical angina' },
            { value: 2, label: 'Atypical angina' },
            { value: 3, label: 'Non-anginal pain' }
          ]},
          { name: 'fasting_blood_sugar', label: 'Fasting Blood Sugar > 120 mg/dL', type: 'toggle', options: ['No', 'Yes'] },
          { name: 'resting_ecg', label: 'Resting ECG Results', type: 'select', options: [
            { value: 0, label: 'Normal' },
            { value: 1, label: 'ST-T wave abnormality' },
            { value: 2, label: 'Left ventricular hypertrophy' }
          ]},
          { name: 'exercise_angina', label: 'Exercise-Induced Angina', type: 'toggle', options: ['No', 'Yes'] },
          { name: 'smoking', label: 'Current Smoker', type: 'toggle', options: ['No', 'Yes'] },
          { name: 'family_history', label: 'Family History of Heart Disease', type: 'toggle', options: ['No', 'Yes'] },
        ]
      }
    ]
  },
  stroke: {
    steps: [
      {
        title: 'Personal Information',
        description: 'Tell us about yourself',
        fields: [
          { name: 'age', label: 'Age', type: 'slider', min: 18, max: 100, unit: 'years' },
          { name: 'gender', label: 'Gender', type: 'toggle', options: ['Male', 'Female'] },
          { name: 'ever_married', label: 'Marital Status', type: 'toggle', options: ['No', 'Yes'] },
        ]
      },
      {
        title: 'Health Profile',
        description: 'Your current health status',
        fields: [
          { name: 'bmi', label: 'Body Mass Index (BMI)', type: 'slider', min: 15, max: 50, step: 0.1, unit: 'kg/m²' },
          { name: 'avg_glucose_level', label: 'Average Glucose Level', type: 'slider', min: 50, max: 300, unit: 'mg/dL' },
          { name: 'hypertension', label: 'Hypertension (High Blood Pressure)', type: 'toggle', options: ['No', 'Yes'] },
          { name: 'heart_disease', label: 'Heart Disease', type: 'toggle', options: ['No', 'Yes'] },
        ]
      },
      {
        title: 'Lifestyle',
        description: 'Your daily life factors',
        fields: [
          { name: 'work_type', label: 'Work Type', type: 'select', options: [
            { value: 'Private', label: 'Private sector' },
            { value: 'Self-employed', label: 'Self-employed' },
            { value: 'Govt_job', label: 'Government job' },
            { value: 'children', label: 'Children/Student' },
            { value: 'Never_worked', label: 'Never worked' }
          ]},
          { name: 'residence_type', label: 'Residence Type', type: 'toggle', options: ['Rural', 'Urban'] },
          { name: 'smoking_status', label: 'Smoking Status', type: 'select', options: [
            { value: 'never smoked', label: 'Never smoked' },
            { value: 'formerly smoked', label: 'Formerly smoked' },
            { value: 'smokes', label: 'Currently smokes' },
            { value: 'Unknown', label: 'Unknown' }
          ]},
          { name: 'physical_activity', label: 'Physical Activity (days/week)', type: 'slider', min: 0, max: 7, unit: 'days' },
          { name: 'alcohol_intake', label: 'Alcohol Intake Level', type: 'slider', min: 0, max: 10, unit: '/10' },
        ]
      }
    ]
  }
}

// Default values for each condition
export const DEFAULT_VALUES = {
  diabetes: {
    age: 45,
    gender: 'Male',
    bmi: 25,
    blood_pressure: 120,
    glucose: 100,
    insulin: 80,
    physical_activity: 3,
    sleep_hours: 7,
    stress_level: 5,
    family_history: 'No',
    smoking: 'No',
    alcohol: 'No'
  },
  heart_disease: {
    age: 50,
    gender: 'Male',
    bmi: 26,
    resting_bp: 120,
    cholesterol: 200,
    max_heart_rate: 150,
    oldpeak: 1.0,
    chest_pain_type: 0,
    fasting_blood_sugar: 'No',
    resting_ecg: 0,
    exercise_angina: 'No',
    smoking: 'No',
    family_history: 'No'
  },
  stroke: {
    age: 50,
    gender: 'Male',
    ever_married: 'Yes',
    bmi: 26,
    avg_glucose_level: 100,
    hypertension: 'No',
    heart_disease: 'No',
    work_type: 'Private',
    residence_type: 'Urban',
    smoking_status: 'never smoked',
    physical_activity: 3,
    alcohol_intake: 2
  }
}

// Risk level configurations
export const RISK_LEVELS = {
  low: {
    label: 'Low Risk',
    color: 'green',
    bgColor: 'bg-green-100',
    textColor: 'text-green-700',
    borderColor: 'border-green-300',
    gradient: 'from-green-400 to-emerald-500',
    icon: '✓',
    message: 'Your risk level is within the normal range. Continue maintaining healthy habits!'
  },
  moderate: {
    label: 'Moderate Risk',
    color: 'orange',
    bgColor: 'bg-amber-100',
    textColor: 'text-amber-700',
    borderColor: 'border-amber-300',
    gradient: 'from-amber-400 to-orange-500',
    icon: '⚠',
    message: 'Some factors indicate elevated risk. Consider lifestyle modifications and consult your doctor.'
  },
  high: {
    label: 'High Risk',
    color: 'red',
    bgColor: 'bg-red-100',
    textColor: 'text-red-700',
    borderColor: 'border-red-300',
    gradient: 'from-red-400 to-rose-500',
    icon: '!',
    message: 'Several factors indicate higher risk. We recommend consulting a healthcare professional soon.'
  },
  critical: {
    label: 'Critical Risk',
    color: 'red',
    bgColor: 'bg-red-200',
    textColor: 'text-red-800',
    borderColor: 'border-red-400',
    gradient: 'from-red-500 to-red-700',
    icon: '!!',
    message: 'Multiple factors indicate significant risk. Please seek medical consultation promptly.'
  }
}
