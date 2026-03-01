/**
 * Compress an image file using canvas to reduce payload size.
 * @param {File} file - The image file to compress
 * @param {number} maxWidth - Maximum width in pixels (default 1200)
 * @param {number} quality - JPEG quality 0-1 (default 0.7)
 * @returns {Promise<string>} - Base64 data URL of compressed image
 */
export function compressImage(file, maxWidth = 1200, quality = 0.7) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const objectUrl = URL.createObjectURL(file)

    img.onload = () => {
      URL.revokeObjectURL(objectUrl)
      const canvas = document.createElement('canvas')
      let { width, height } = img

      if (width > maxWidth) {
        height = Math.round((height * maxWidth) / width)
        width = maxWidth
      }

      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, width, height)

      resolve(canvas.toDataURL('image/jpeg', quality))
    }

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      // Fallback: read as original data URL
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target.result)
      reader.onerror = reject
      reader.readAsDataURL(file)
    }

    img.src = objectUrl
  })
}

/**
 * Validate an image file's type and size
 * @param {File} file
 * @param {number} maxSizeMB
 * @returns {{ valid: boolean, error?: string }}
 */
export function validateImageFile(file, maxSizeMB = 10) {
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/heic']
  const ALLOWED_EXT = /\.(jpg|jpeg|png|webp|heic)$/i

  if (!file) return { valid: false, error: 'No file selected' }

  if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(ALLOWED_EXT)) {
    return { valid: false, error: 'Please select a valid image file (JPG, PNG, or WebP)' }
  }

  if (file.size > maxSizeMB * 1024 * 1024) {
    return { valid: false, error: `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum is ${maxSizeMB}MB.` }
  }

  return { valid: true }
}
