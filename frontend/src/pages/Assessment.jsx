import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowLeft, ArrowRight, Loader2 } from 'lucide-react'
import Layout from '../components/Layout'
import ProgressBar from '../components/ProgressBar'
import FormField from '../components/FormField'
import LoadingOverlay from '../components/LoadingOverlay'
import { useAssessment } from '../context/AssessmentContext'
import { CONDITIONS, FORM_CONFIGS, DEFAULT_VALUES } from '../config/conditions'
import healthAPI from '../services/api'

export default function Assessment() {
  const { condition } = useParams()
  const navigate = useNavigate()
  const { setSelectedCondition, setFormData, setResults, setIsLoading, isLoading } = useAssessment()

  const [currentStep, setCurrentStep] = useState(0)
  const [formValues, setFormValues] = useState({})
  const [errors, setErrors] = useState({})

  const conditionConfig = CONDITIONS[condition]
  const formConfig = FORM_CONFIGS[condition]
  const totalSteps = formConfig?.steps?.length || 1

  useEffect(() => {
    if (!conditionConfig || !formConfig) {
      navigate('/select')
      return
    }
    setSelectedCondition(condition)
    setFormValues(DEFAULT_VALUES[condition] || {})
  }, [condition])

  const handleFieldChange = (name, value) => {
    setFormValues(prev => ({ ...prev, [name]: value }))
    // Clear error when field changes
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }))
    }
  }

  const validateStep = () => {
    const currentFields = formConfig.steps[currentStep].fields
    const newErrors = {}
    
    currentFields.forEach(field => {
      if (formValues[field.name] === undefined || formValues[field.name] === '') {
        newErrors[field.name] = 'This field is required'
      }
    })

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleNext = () => {
    if (validateStep()) {
      if (currentStep < totalSteps - 1) {
        setCurrentStep(prev => prev + 1)
      } else {
        handleSubmit()
      }
    }
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1)
    } else {
      navigate('/select')
    }
  }

  const prepareSubmitData = () => {
    const data = { ...formValues }
    
    // Convert values for API
    Object.keys(data).forEach(key => {
      // Convert Yes/No toggles to 1/0 (except specific string fields)
      if (data[key] === 'Yes' && key !== 'ever_married') data[key] = 1
      else if (data[key] === 'No' && key !== 'ever_married') data[key] = 0
      
      // Convert gender to lowercase for API
      if (key === 'gender') {
        data[key] = data[key].toLowerCase()
      }
      
      // Convert residence_type toggle values
      if (key === 'residence_type') {
        // Keep as string - API expects 'Urban' or 'Rural'
      }
    })
    
    // Heart disease: chest_pain_type needs to be 1-4 (API schema), but form uses 0-3
    if (condition === 'heart_disease' && data.chest_pain_type !== undefined) {
      data.chest_pain_type = parseInt(data.chest_pain_type) + 1
    }

    return data
  }

  const handleSubmit = async () => {
    setIsLoading(true)
    setFormData(formValues)

    try {
      const submitData = prepareSubmitData()
      const response = await healthAPI.predict(condition, submitData)
      
      setResults({
        ...response,
        condition: condition,
        submittedData: formValues
      })
      
      navigate('/results')
    } catch (error) {
      console.error('Prediction error:', error)
      // Extract error message properly
      let errorMessage = 'Unable to process your assessment. Please try again.'
      if (typeof error === 'string') {
        errorMessage = error
      } else if (error?.message) {
        errorMessage = error.message
      } else if (error?.detail) {
        errorMessage = error.detail
      }
      // Still navigate to results with error state
      setResults({
        error: errorMessage,
        condition: condition,
        submittedData: formValues
      })
      navigate('/results')
    } finally {
      setIsLoading(false)
    }
  }

  if (!conditionConfig || !formConfig) {
    return null
  }

  const currentStepConfig = formConfig.steps[currentStep]

  return (
    <Layout>
      {isLoading && <LoadingOverlay />}
      
      <div className="min-h-screen py-8 px-4">
        <div className="max-w-2xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            {/* Back Button */}
            <button
              onClick={handleBack}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>{currentStep === 0 ? 'Change Condition' : 'Previous Step'}</span>
            </button>

            {/* Condition Badge */}
            <div className="flex items-center gap-3 mb-4">
              <span className="text-3xl">{conditionConfig.icon}</span>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{conditionConfig.name} Assessment</h1>
                <p className="text-gray-500">{conditionConfig.subtitle}</p>
              </div>
            </div>

            {/* Progress Bar */}
            <ProgressBar current={currentStep + 1} total={totalSteps} />
          </motion.div>

          {/* Form Card */}
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="card"
            >
              {/* Step Header */}
              <div className="mb-8">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  {currentStepConfig.title}
                </h2>
                <p className="text-gray-500">
                  {currentStepConfig.description}
                </p>
              </div>

              {/* Form Fields */}
              <div className="space-y-6">
                {currentStepConfig.fields.map((field) => (
                  <FormField
                    key={field.name}
                    field={field}
                    value={formValues[field.name]}
                    onChange={(value) => handleFieldChange(field.name, value)}
                    error={errors[field.name]}
                  />
                ))}
              </div>

              {/* Navigation Buttons */}
              <div className="flex justify-between mt-10 pt-6 border-t border-gray-100">
                <button
                  onClick={handleBack}
                  className="btn-secondary"
                >
                  Back
                </button>
                <button
                  onClick={handleNext}
                  className="btn-primary flex items-center gap-2"
                >
                  {currentStep === totalSteps - 1 ? (
                    <>
                      Get Results
                      {isLoading ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <ArrowRight className="w-5 h-5" />
                      )}
                    </>
                  ) : (
                    <>
                      Continue
                      <ArrowRight className="w-5 h-5" />
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Tips */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mt-6 p-4 rounded-xl bg-blue-50 border border-blue-100"
          >
            <p className="text-sm text-blue-700">
              <span className="font-medium">💡 Tip:</span> For the most accurate results, 
              use recent health measurements from your doctor or home monitoring devices.
            </p>
          </motion.div>
        </div>
      </div>
    </Layout>
  )
}
