"use client";

import * as Sentry from "@sentry/nextjs";

Sentry.init({
    dsn: "https://0d0b514ea6edd21fc3616f5919196888@o4510350699659264.ingest.us.sentry.io/4510641231757312",

    // Set tracesSampleRate to 1.0 to capture 100%
    // of transactions for performance monitoring.
    // We recommend adjusting this value in production
    tracesSampleRate: 1.0,

    integrations: [Sentry.browserTracingIntegration()],

    // Set tracePropagationTargets to localhost and the API URL
    tracePropagationTargets: ["localhost", /^http:\/\/localhost:8000/, /^https:\/\/yourserver\.io\/api/],
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
