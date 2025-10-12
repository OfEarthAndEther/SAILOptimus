import React from "react";
import SimulationIframe from "@/components/SimulationIframe";

export default function Simulation() {
  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Vessel Simulation</h1>
        <p className="text-muted-foreground mt-2">
          Optimize cargo splits and dispatch plans to minimize total logistics spend
        </p>
      </div>
      
      <SimulationIframe className="w-full" />
    </div>
  );
}
