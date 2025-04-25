# PanicSense Render Deployment Fix

## 502 Bad Gateway Error Solution

If you're getting a 502 Bad Gateway error when accessing your PanicSense deployment on Render, it's likely due to one of these issues:

1. **Database Schema Mismatch**: The `created_at` column is missing or named differently in your database tables.
2. **Python Service Not Starting**: The Python daemon failed to start due to missing dependencies or permissions.
3. **WebSocket Connection Issues**: WebSocket connections are failing, preventing real-time updates.

## How This Package Solves These Issues

The `production-server-fixed.cjs` file included in this package contains special code to detect and adapt to these issues:

### 1. Schema Adaptation:

The server now checks each table's schema before querying and adapts the SQL query to match the available columns:

```javascript
// First check the table schema to avoid errors
const schemaQuery = await client.query(`
  SELECT column_name 
  FROM information_schema.columns 
  WHERE table_name = 'disaster_events'
`);

const columns = schemaQuery.rows.map(row => row.column_name);
console.log('Disaster events columns:', columns);

// Use different queries based on available columns
let result;
if (columns.includes('created_at')) {
  result = await client.query('SELECT * FROM disaster_events ORDER BY created_at DESC');
} else if (columns.includes('timestamp')) {
  result = await client.query('SELECT * FROM disaster_events ORDER BY timestamp DESC');
} else {
  result = await client.query('SELECT * FROM disaster_events');
}
```

### 2. Python Service Auto-detection and Restart:

The server uses an intelligent system to find the correct Python executable and automatically restarts the Python service if it crashes:

```javascript
function findPythonExecutable() {
  // Try different Python executable names
  const possiblePythons = ['python', 'python3', 'python3.11', 'python3.10', 'python3.9'];
  
  for (const pythonName of possiblePythons) {
    try {
      const result = require('child_process').spawnSync(pythonName, ['--version']);
      if (result.status === 0) {
        console.log(`Found Python executable: ${pythonName}`);
        return pythonName;
      }
    } catch (error) {
      debug(`Python executable ${pythonName} not available`);
    }
  }
  
  // If we get here, default to 'python'
  console.log('No Python executable found, defaulting to "python"');
  return 'python';
}
```

### 3. WebSocket Support:

Built-in WebSocket server for real-time communication:

```javascript
// Create WebSocket server for real-time updates
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  // Send initial data upon connection
  ws.send(JSON.stringify({
    type: 'connection_established',
    timestamp: new Date().toISOString(),
    pythonActive: pythonProcess !== null,
    recentEvents: pythonEvents.slice(-5),
    recentErrors: pythonErrors.slice(-5)
  }));
});
```

## Environment Variables

For optimal operation, set these environment variables in Render:

- `DATABASE_URL`: Your PostgreSQL database URL
- `NODE_ENV`: Set to `production`
- `SESSION_SECRET`: Any random secure string
- `DEBUG`: Set to `true` for detailed logs (optional)

## TROUBLESHOOTING

If you're still experiencing issues:

1. **Check Logs**: Go to your Render dashboard and check the logs for error messages.
2. **Python Logs**: Access `/api/python-logs` endpoint to see Python-specific errors.
3. **Database Tables**: Make sure your database tables have the expected structure.
4. **Restart Service**: Sometimes simply restarting the service in Render dashboard resolves temporary issues.

## Need Help?

If you continue to experience issues, create a detailed error report with:
1. Complete error messages from the Render logs
2. Screenshots of the 502 error page
3. Database schema information

The most frequent issues are related to database schema discrepancies or Python execution problems, which this package has been specifically designed to handle automatically.