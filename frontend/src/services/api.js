const API_BASE_URL = 'http://127.0.0.1:8000'

export const healthAPI = {
  /**
   * Check API health status
   */
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`)
      return response.ok
    } catch {
      return false
    }
  },

  /**
   * Submit health assessment for prediction
   * @param {string} condition - 'diabetes' | 'heart_disease' | 'stroke'
   * @param {object} data - Form data for the condition
   */
  async predict(condition, data) {
    const endpoint = `${API_BASE_URL}/predict/${condition}`
    
    console.log('Sending prediction request:', { endpoint, data })
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        const errorData = await response.json()
        console.error('API Error Response:', errorData)
        
        // Extract error message from various formats
        let errorMessage = 'Prediction failed'
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail
        } else if (Array.isArray(errorData.detail)) {
          // Pydantic validation errors come as array
          errorMessage = errorData.detail.map(e => e.msg || e.message).join(', ')
        } else if (errorData.message) {
          errorMessage = errorData.message
        }
        
        throw new Error(errorMessage)
      }

      return await response.json()
    } catch (error) {
      console.error('API Error:', error)
      throw error
    }
  },

  /**
   * Get diabetes prediction
   */
  async predictDiabetes(data) {
    return this.predict('diabetes', data)
  },

  /**
   * Get heart disease prediction
   */
  async predictHeartDisease(data) {
    return this.predict('heart_disease', data)
  },

  /**
   * Get stroke prediction
   */
  async predictStroke(data) {
    return this.predict('stroke', data)
  }
}

export default healthAPI
