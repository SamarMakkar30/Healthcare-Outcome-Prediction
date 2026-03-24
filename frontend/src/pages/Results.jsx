import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, RefreshCw, Home, AlertTriangle } from 'lucide-react'
import Layout from '../components/Layout'
import RiskMeter from '../components/RiskMeter'
import RiskFactorCard from '../components/RiskFactorCard'
import RecommendationCard from '../components/RecommendationCard'
import { useAssessment } from '../context/AssessmentContext'
import { CONDITIONS, RISK_LEVELS } from '../config/conditions'

export default function Results() {
  const navigate = useNavigate()
  const { results, selectedCondition, resetAssessment } = useAssessment()

  useEffect(() => {
    if (!results) {
      navigate('/select')
    }
  }, [results, navigate])

  if (!results) return null

  const condition = CONDITIONS[results.condition || selectedCondition]
  const hasError = results.error

  // Normalize risk level from API response
  const getRiskLevel = () => {
    const level = results.risk_level?.toLowerCase() || 'moderate'
    return RISK_LEVELS[level] || RISK_LEVELS.moderate
  }

  const riskLevel = getRiskLevel()
  // API returns risk_probability (0-1), convert to percentage
  const probability = (results.risk_probability ?? results.probability ?? 0) * 100

  const handleStartOver = () => {
    resetAssessment()
    navigate('/select')
  }

  const handleGoHome = () => {
    resetAssessment()
    navigate('/')
  }

  if (hasError) {
    return (
      <Layout>
        <div className="min-h-screen py-8 px-4 flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-md w-full card text-center"
          >
            <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-6">
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
            <h2 className="text-xl font-bold text-gray-900 mb-2">Unable to Process</h2>
            <p className="text-gray-600 mb-6">{results.error}</p>
            <div className="flex gap-3 justify-center">
              <button onClick={handleStartOver} className="btn-secondary flex items-center gap-2">
                <RefreshCw className="w-4 h-4" />
                Try Again
              </button>
              <button onClick={handleGoHome} className="btn-primary flex items-center gap-2">
                <Home className="w-4 h-4" />
                Home
              </button>
            </div>
          </motion.div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="min-h-screen py-8 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between mb-8"
          >
            <button
              onClick={handleStartOver}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>New Assessment</span>
            </button>
            <div className="flex items-center gap-3">
              <span className="text-2xl">{condition?.icon}</span>
              <span className="font-semibold text-gray-700">{condition?.name} Results</span>
            </div>
          </motion.div>

          {/* Main Results Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card mb-6"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Left: Risk Meter */}
              <div className="flex flex-col items-center justify-center">
                <h2 className="text-lg font-semibold text-gray-700 mb-6">Your Risk Score</h2>
                <RiskMeter 
                  percentage={Math.round(probability)} 
                  riskLevel={riskLevel}
                />
              </div>

              {/* Right: Risk Level Info */}
              <div className="flex flex-col justify-center">
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${riskLevel.bgColor} ${riskLevel.textColor} font-semibold mb-4 w-fit`}>
                  <span className="text-lg">{riskLevel.icon}</span>
                  <span>{riskLevel.label}</span>
                </div>
                <p className="text-gray-600 mb-6">{riskLevel.message}</p>
                
                {/* Key Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-gray-50">
                    <p className="text-sm text-gray-500 mb-1">Risk Probability</p>
                    <p className="text-2xl font-bold text-gray-900">{Math.round(probability)}%</p>
                  </div>
                  <div className="p-4 rounded-xl bg-gray-50">
                    <p className="text-sm text-gray-500 mb-1">Confidence</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {results.confidence_interval ? 'High' : 'Standard'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Risk Factors */}
          {(results.top_risk_factors?.length > 0 || results.contributing_factors?.length > 0) && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-6"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Risk Drivers</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* top_risk_factors is array of RiskFactor objects: {feature, value, impact, importance} */}
                {results.top_risk_factors?.slice(0, 6).map((factor, index) => (
                  <RiskFactorCard key={index} factor={factor} index={index} />
                ))}
                
                {/* contributing_factors is array of strings - convert to factor format */}
                {(!results.top_risk_factors || results.top_risk_factors.length === 0) && 
                  results.contributing_factors?.map((factorName, index) => (
                    <RiskFactorCard 
                      key={`contrib-${index}`} 
                      factor={{ 
                        feature: factorName, 
                        name: factorName,
                        impact: 'increases', 
                        importance: 0.5 
                      }} 
                      index={index} 
                    />
                  ))
                }
              </div>
            </motion.div>
          )}

          {/* Recommendations */}
          {results.recommendations && results.recommendations.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mb-6"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Personalized Recommendations</h3>
              <div className="space-y-4">
                {results.recommendations.map((rec, index) => (
                  <RecommendationCard key={index} recommendation={rec} index={index} />
                ))}
              </div>
            </motion.div>
          )}

          {/* Action Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="card bg-gradient-to-r from-teal-50 to-blue-50 border border-teal-100"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">What You Can Do Next</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ActionCard
                icon="🩺"
                title="Consult a Doctor"
                description="Share these results with your healthcare provider"
              />
              <ActionCard
                icon="📋"
                title="Track Your Health"
                description="Monitor the key metrics identified above"
              />
              <ActionCard
                icon="🎯"
                title="Take Action"
                description="Start with the top recommendations"
              />
            </div>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-6 p-4 rounded-xl bg-amber-50 border border-amber-200"
          >
            <p className="text-sm text-amber-800">
              <span className="font-semibold">⚠️ Important:</span> This assessment provides an estimate based on 
              the information you provided. It is not a medical diagnosis. Please consult with a qualified 
              healthcare professional for proper medical advice, diagnosis, or treatment.
            </p>
          </motion.div>

          {/* Bottom Actions */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="flex justify-center gap-4 mt-8"
          >
            <button onClick={handleStartOver} className="btn-secondary flex items-center gap-2">
              <RefreshCw className="w-4 h-4" />
              New Assessment
            </button>
            <button onClick={handleGoHome} className="btn-primary flex items-center gap-2">
              <Home className="w-4 h-4" />
              Back to Home
            </button>
          </motion.div>
        </div>
      </div>
    </Layout>
  )
}

function ActionCard({ icon, title, description }) {
  return (
    <div className="p-4 rounded-xl bg-white/70 backdrop-blur-sm">
      <span className="text-2xl">{icon}</span>
      <h4 className="font-semibold text-gray-900 mt-2">{title}</h4>
      <p className="text-sm text-gray-600 mt-1">{description}</p>
    </div>
  )
}
