import * as Sentry from '@sentry/nextjs';

export async function register() {
    if (process.env.NEXT_RUNTIME === 'nodejs') {
        Sentry.init({
            dsn: "https://0d0b514ea6edd21fc3616f5919196888@o4510350699659264.ingest.us.sentry.io/4510641231757312",
            tracesSampleRate: 1.0,
            debug: false,
        });
    }

    if (process.env.NEXT_RUNTIME === 'edge') {
        Sentry.init({
            dsn: "https://0d0b514ea6edd21fc3616f5919196888@o4510350699659264.ingest.us.sentry.io/4510641231757312",
            tracesSampleRate: 1.0,
            debug: false,
        });
    }
}

export const onRequestError = Sentry.captureRequestError;
