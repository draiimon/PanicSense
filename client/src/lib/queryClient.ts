import { QueryClient, QueryFunction } from "@tanstack/react-query";

/**
 * Enhanced error handling with detailed error response extraction
 */
async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    try {
      // Try to parse as JSON first
      const contentType = res.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        const errorData = await res.json();
        const errorMessage = errorData.message || errorData.error || JSON.stringify(errorData);
        throw new Error(`${res.status}: ${errorMessage}`);
      } else {
        // Fallback to text
        const text = (await res.text()) || res.statusText;
        throw new Error(`${res.status}: ${text}`);
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      // If JSON parsing fails, fallback to status text
      throw new Error(`${res.status}: ${res.statusText}`);
    }
  }
}

/**
 * Centralized API request function with improved error handling
 */
export async function apiRequest<T>(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: {
      ...(data ? { "Content-Type": "application/json" } : {}),
      // Add request ID for tracing
      "X-Request-ID": `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
    },
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);
  
  // For DELETE or empty responses
  if (method === "DELETE" || res.status === 204) {
    return {} as T;
  }
  
  return await res.json();
}

type UnauthorizedBehavior = "returnNull" | "throw";

/**
 * Enhanced query function with better error handling
 */
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    // Add retries for network failures
    const maxRetries = 2;
    let retryCount = 0;
    let lastError: Error | null = null;

    while (retryCount <= maxRetries) {
      try {
        const res = await fetch(queryKey[0] as string, {
          credentials: "include",
          headers: {
            // Add performance tracking header
            "X-Request-Start": Date.now().toString(),
          }
        });

        if (unauthorizedBehavior === "returnNull" && res.status === 401) {
          return null;
        }

        await throwIfResNotOk(res);
        return await res.json();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        // Only retry network errors, not API errors
        if (lastError.message.includes('Failed to fetch') || 
            lastError.message.includes('NetworkError')) {
          retryCount++;
          // Exponential backoff
          await new Promise(r => setTimeout(r, 2 ** retryCount * 100));
          continue;
        }
        
        throw lastError;
      }
    }
    
    throw lastError!;
  };

/**
 * Enhanced QueryClient with improved performance settings and error handling
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes instead of Infinity to improve cache freshness
      retry: (failureCount, error) => {
        // Don't retry for client errors (4xx)
        if (error instanceof Error && error.message.match(/^4\d\d:/)) {
          return false;
        }
        return failureCount < 2; // Retry server errors twice
      },
      gcTime: 10 * 60 * 1000, // 10 minutes garbage collection time (was cacheTime in v4)
      placeholderData: 'keepPrevious', // Keep previous data while fetching (replaces keepPreviousData)
    },
    mutations: {
      // No retries for mutations to prevent duplicate operations
      retry: false,
    },
  },
});
