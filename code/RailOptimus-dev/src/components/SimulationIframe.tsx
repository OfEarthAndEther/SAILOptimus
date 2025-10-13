import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, ExternalLink, RefreshCw, AlertCircle } from "lucide-react";

interface SimulationIframeProps {
  className?: string;
}

const SimulationIframe: React.FC<SimulationIframeProps> = ({ className }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);

  // SAILoptimisation Dash app runs by default on port 5006 (see app.run_server in project)
  const SIMULATION_URL = "http://localhost:5006";

  const handleIframeLoad = () => {
    setIsLoading(false);
    setIsError(false);
  };

  const handleIframeError = () => {
    setIsLoading(false);
    setIsError(true);
  };

  const refreshSimulation = () => {
    setIsLoading(true);
    setIsError(false);
    setIframeKey(prev => prev + 1);
  };

  // Lightweight reachability check: try to load the root favicon (fast) to detect if
  // the local Dash server is reachable. This avoids CORS issues that occur when
  // trying to fetch HTML from another origin.
  const checkReachable = () => {
    setIsLoading(true);
    setIsError(false);

    try {
      const img = new Image();
      // append cache-buster so refresh works
      img.src = `${SIMULATION_URL}/_favicon.ico?_cb=${Date.now()}`;
      const onSuccess = () => {
        setIsLoading(false);
        setIsError(false);
        cleanup();
      };
      const onFail = () => {
        setIsLoading(false);
        setIsError(true);
        cleanup();
      };
      const cleanup = () => {
        img.onload = null;
        img.onerror = null;
      };
      img.onload = onSuccess;
      img.onerror = onFail;

      // fallback timeout in case the request hangs
      setTimeout(() => {
        if (isLoading) {
          setIsLoading(false);
          setIsError(true);
        }
      }, 4000);
    } catch (_e) {
      setIsLoading(false);
      setIsError(true);
    }
  };

  const openInNewTab = () => {
    window.open(SIMULATION_URL, '_blank');
  };

  return (
    <div className={className}>
      <Card className="h-full">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <CardTitle className="flex items-center gap-2">
            <span>Optimizer Window</span>
            {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => { checkReachable(); refreshSimulation(); }}
              disabled={isLoading}
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={openInNewTab}
            >
              <ExternalLink className="h-4 w-4 mr-1" />
              Open in New Tab
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {isError ? (
            <Alert className="m-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="flex flex-col gap-2">
                <span>
                  Failed to load Simulation. Please ensure the SAILoptimisation app is running on port 5006 and is reachable (CORS allowed).
                </span>
                <div className="flex gap-2">
                  <Button size="sm" onClick={refreshSimulation}>
                    Retry
                  </Button>
                  <Button size="sm" variant="outline" onClick={openInNewTab}>
                    Open Directly
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <div className="relative">
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-8 w-8 animate-spin" />
                    <p className="text-sm text-muted-foreground">Loading...</p>
                  </div>
                </div>
              )}
              <iframe
                key={iframeKey}
                src={SIMULATION_URL}
                title="Vessel Logistic Tracking Simulation"
                width="100%"
                height="800px"
                style={{
                  border: 'none',
                  borderRadius: '0 0 8px 8px',
                  minHeight: '800px'
                }}
                onLoad={handleIframeLoad}
                onError={handleIframeError}
                allow="fullscreen"
                // Relax sandboxing so cross-origin Dash app can run interactive scripts.
                // keep allow-same-origin and allow-scripts; allow-popups kept for open-in-new-tab
                sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default SimulationIframe;