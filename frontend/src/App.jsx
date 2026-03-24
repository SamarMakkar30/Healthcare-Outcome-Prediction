import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'

// Pages
import Landing from './pages/Landing'
import SelectCondition from './pages/SelectCondition'
import Assessment from './pages/Assessment'
import Results from './pages/Results'
import NotFound from './pages/NotFound'

// Context
import { AssessmentProvider } from './context/AssessmentContext'

function App() {
  return (
    <AssessmentProvider>
      <div className="min-h-screen">
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/select" element={<SelectCondition />} />
            <Route path="/assess/:condition" element={<Assessment />} />
            <Route path="/results" element={<Results />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AnimatePresence>
      </div>
    </AssessmentProvider>
  )
}

export default App
