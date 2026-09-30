import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  outputFileTracingRoot: path.join(__dirname),
  // Chapters are loaded with fs.readdirSync, which the file tracer cannot see.
  outputFileTracingIncludes: {
    "/*": ["./volumes/**/*"],
    "/m/*": ["./volumes/**/*"],
    "/m/*/*": ["./volumes/**/*"],
  },
};

export default nextConfig;
