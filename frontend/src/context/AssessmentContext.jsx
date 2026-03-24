import { createContext, useContext, useState } from 'react'

const AssessmentContext = createContext()

export function AssessmentProvider({ children }) {
  const [selectedCondition, setSelectedCondition] = useState(null)
  const [formData, setFormData] = useState({})
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const resetAssessment = () => {
    setSelectedCondition(null)
    setFormData({})
    setResults(null)
    setIsLoading(false)
  }

  const value = {
    selectedCondition,
    setSelectedCondition,
    formData,
    setFormData,
    results,
    setResults,
    isLoading,
    setIsLoading,
    resetAssessment
  }

  return (
    <AssessmentContext.Provider value={value}>
      {children}
    </AssessmentContext.Provider>
  )
}

export function useAssessment() {
  const context = useContext(AssessmentContext)
  if (!context) {
    throw new Error('useAssessment must be used within AssessmentProvider')
  }
  return context
}
