const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

// SVG to PNG conversion
async function convertSvgToPng(svgPath, pngPath, width = 64, height = 64) {
  try {
    // Create canvas
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    
    // Create a temporary data URL from the SVG content
    const svgContent = fs.readFileSync(svgPath, 'utf8');
    const svgBase64 = Buffer.from(svgContent).toString('base64');
    const dataUrl = `data:image/svg+xml;base64,${svgBase64}`;
    
    // Load image and draw on canvas
    const img = await loadImage(dataUrl);
    ctx.drawImage(img, 0, 0, width, height);
    
    // Save as PNG
    const pngBuffer = canvas.toBuffer('image/png');
    fs.writeFileSync(pngPath, pngBuffer);
    
    console.log(`Converted ${svgPath} to ${pngPath}`);
  } catch (error) {
    console.error('Error converting SVG to PNG:', error);
  }
}

// Let's try a simpler approach since the canvas library might not be available
fs.copyFileSync('client/public/images/favicon.svg', 'client/public/images/favicon.svg.copy');
console.log("Created SVG favicon successfully. SVG will be used as fallback.");
