const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

async function createRoundIcon() {
  try {
    // Input file path
    const inputFile = path.join('attached_assets', '{E420C08C-1924-4628-800E-FD13D3EC8F4F}_1743045522863.png');
    
    // Output file paths
    const outputFile = path.join('client', 'public', 'favicon.png');
    const outputFileImages = path.join('client', 'public', 'images', 'panicsense-icon.png');
    
    // Create a circular mask
    const size = 64;
    const circleBuffer = Buffer.from(
      `<svg><circle cx="${size/2}" cy="${size/2}" r="${size/2}" /></svg>`
    );

    // Process the image - resize and apply circular mask
    await sharp(inputFile)
      .resize(size, size)
      .composite([{
        input: circleBuffer,
        blend: 'dest-in'
      }])
      .toFile(outputFile);
    
    // Copy to images directory as well
    fs.copyFileSync(outputFile, outputFileImages);
    
    console.log('Round icon created successfully!');
    return true;
  } catch (error) {
    console.error('Error creating round icon:', error);
    return false;
  }
}

// Execute the function
createRoundIcon();
