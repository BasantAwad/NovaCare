const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverActions: {
      allowedOrigins: ["10.174.134.241:3000", "10.174.134.247:3000"]
    }
  },
  async rewrites() {
    return [
      {
        source: '/api/novabot/:path*',
        destination: 'http://127.0.0.1:5000/api/:path*'
      },
      {
        source: '/api/auth/:path*',
        destination: 'http://127.0.0.1:5001/api/auth/:path*'
      }
    ];
  }
};

export default nextConfig;
