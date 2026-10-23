/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",

  // Allow the browser to make cross-origin requests directly to the HF Space backend
  // (needed for large file uploads that bypass Vercel's 4.5 MB payload limit)
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Access-Control-Allow-Methods", value: "GET,POST,OPTIONS" },
          { key: "Access-Control-Allow-Headers", value: "Content-Type" },
        ],
      },
    ];
  },

  // Increase Next.js API route body size limit to 50 MB for video uploads
  // (only applies to the /api/* proxy routes, not direct browser → HF uploads)
  experimental: {
    serverActions: {
      bodySizeLimit: "50mb",
    },
  },
};

module.exports = nextConfig;
