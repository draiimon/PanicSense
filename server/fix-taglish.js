/**
 * Server-side modification of incoming CSV data to preserve Taglish language entries
 * 
 * This solution adds a post-processing step when receiving CSV data 
 * to preserve Taglish entries
 */

// Ensure Taglish is treated as a valid language
export function preserveTaglishEntries(results) {
  if (!results || !Array.isArray(results)) return results;
  
  // Process each entry to ensure Taglish is respected
  return results.map(entry => {
    // If a record is labeled as Taglish in CSV, keep it
    if (entry.language && entry.language.toLowerCase() === 'taglish') {
      entry.language = 'Taglish'; // Ensure proper capitalization
    }
    return entry;
  });
}