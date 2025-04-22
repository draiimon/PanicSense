// This is a compatibility wrapper for the index.ts file
// It allows running the server even in environments where TS files cannot be directly executed

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('Starting server via compatibility wrapper...');

// Use dynamic import for the TypeScript module
try {
  const indexModule = await import('./index-wrapper.ts');
  console.log('Server started successfully');
} catch (error) {
  console.error('Error starting server:', error);
  process.exit(1);
}