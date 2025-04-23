# PanicSense - Render Deployment

## Deployment sa Render.com

### Step 1: Setup sa Render Dashboard

Sa Render dashboard, i-add ang new web service at i-point ito sa GitHub repository mo.

### Step 2: Configure ang deployment

#### OPTION 1: Gamit ang package.json scripts (RECOMMENDED)

**Build Command:**
```
npm run build
```

**Start Command:**
```
npm run start
```

#### OPTION 2: Gamit ang run.cjs (Alternative/Fallback)

**Build Command:**
```
npm install
```

**Start Command:**
```
node run.cjs
```

### Step 3: Environment Variables

I-add ang mga sumusunod na environment variables:

- `NODE_ENV` = `production`
- `DATABASE_URL` = [Your PostgreSQL URL]
- `GROQ_API_KEY` = [Your Groq API Key]

## Paano Gumagana Ang Deployment?

### Option 1: package.json scripts (RECOMMENDED)
Ginagamit nito ang sarili mong scripts na nakadeclare sa package.json:

- **build**: Nagbi-build ng client React application at server TypeScript files
- **start**: Pinapatakbo ang application sa production mode

Advantages:
- Full build process with proper optimization
- TypeScript compilation with proper type-checking
- Cleaner deployment workflow

### Option 2: run.cjs (Simple Fallback)
Simpleng file na nagko-connect sa main application:

- CommonJS format para compatible kahit may "type": "module" sa package.json 
- Direkta na tinatawag ang main index.js para patakbuhin ang application
- .cjs extension para masabi kay Node.js na iba ito sa ES modules

Lahat ng files ay makikita ni Render, walang hidden files o complex setup scripts!