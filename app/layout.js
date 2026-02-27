import { Analytics } from '@vercel/analytics/react';

export const metadata = {
  title: 'Train My Own GPT',
  description: 'Build, train, and run a GPT from scratch — entirely in your browser',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body style={{ margin: 0, padding: 0 }}>{children}<Analytics /></body>
    </html>
  );
}
