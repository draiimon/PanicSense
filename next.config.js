/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  distDir: 'dist/.next',
  experimental: {
    outputFileTracingRoot: process.cwd(),
  },
  // Use custom publicRuntimeConfig for shared variables
  publicRuntimeConfig: {
    apiBase: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000',
  },
  // Ensure Next.js respects the API routes from Express
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '/api/:path*',
      },
      {
        source: '/ws',
        destination: '/ws',
      }
    ];
  },
};

module.exports = nextConfig;