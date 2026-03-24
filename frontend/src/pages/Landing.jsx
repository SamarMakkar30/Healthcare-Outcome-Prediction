import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Heart, Shield, Activity, ArrowRight, CheckCircle } from 'lucide-react'
import Layout from '../components/Layout'

export default function Landing() {
  const navigate = useNavigate()

  const features = [
    {
      icon: <Shield className="w-6 h-6" />,
      title: 'Private & Secure',
      description: 'Your health data stays on your device'
    },
    {
      icon: <Activity className="w-6 h-6" />,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning models'
    },
    {
      icon: <CheckCircle className="w-6 h-6" />,
      title: 'Instant Results',
      description: 'Get insights in seconds'
    }
  ]

  return (
    <Layout>
      <div className="min-h-screen flex flex-col">
        {/* Hero Section */}
        <div className="flex-1 flex items-center justify-center px-4 py-12">
          <div className="max-w-4xl mx-auto text-center">
            {/* Floating Heart Icon */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-8"
            >
              <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 shadow-2xl shadow-teal-500/30 animate-float">
                <Heart className="w-12 h-12 text-white" />
              </div>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-6"
            >
              AI Health Risk
              <span className="block gradient-text">Assessment</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto leading-relaxed"
            >
              Understand your health risks for diabetes, heart disease, and stroke 
              with our AI-powered assessment tool. Take the first step toward a healthier future.
            </motion.p>

            {/* CTA Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <button
                onClick={() => navigate('/select')}
                className="group btn-primary inline-flex items-center gap-3 text-lg px-8 py-4"
              >
                Start Assessment
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </motion.div>

            {/* Features */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6"
            >
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="flex flex-col items-center p-6 rounded-2xl bg-white/60 backdrop-blur-sm border border-gray-100"
                >
                  <div className="w-12 h-12 rounded-xl bg-teal-100 flex items-center justify-center text-teal-600 mb-4">
                    {feature.icon}
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-1">{feature.title}</h3>
                  <p className="text-sm text-gray-500">{feature.description}</p>
                </div>
              ))}
            </motion.div>
          </div>
        </div>

        {/* Disclaimer Banner */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="bg-gradient-to-r from-amber-50 to-orange-50 border-t border-amber-200"
        >
          <div className="max-w-4xl mx-auto px-4 py-4">
            <p className="text-sm text-amber-800 text-center">
              <span className="font-semibold">⚠️ Important Disclaimer:</span>{' '}
              This tool provides educational health risk estimates only. It is not a medical diagnosis. 
              Always consult with a qualified healthcare professional for medical advice.
            </p>
          </div>
        </motion.div>
      </div>
    </Layout>
  )
}
