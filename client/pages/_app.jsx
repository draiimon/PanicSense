import '../src/index.css';

// This is a simple Next.js wrapper around the existing React app
export default function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />;
}