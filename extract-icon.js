import sharp from 'sharp';
import fs from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

// The icon is in the top-left corner of the image
// We'll extract a 40x40 pixel area from the coordinates (35, 32)
async function extractIcon() {
  try {
    // This is just based on visual inspection of the image
    await sharp('./attached_assets/{0D003942-1B94-4D6D-92A7-4F71BCC9AAE6}_1745105701879.png')
      .extract({ left: 25, top: 22, width: 60, height: 60 })
      .toFile('./temp_extract/icon_extract.png');
    
    console.log('Icon extracted successfully!');

    // Now copy the extracted icon to be the new favicon
    fs.copyFileSync('./temp_extract/icon_extract.png', './client/public/favicon.png');
    console.log('Favicon updated successfully!');
  } catch (error) {
    console.error('Error extracting icon:', error);
  }
}

extractIcon();