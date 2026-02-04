import { useEffect, useState } from 'react';
import { LuCloudSun, LuDroplets } from 'react-icons/lu';
import Card from '../Card';

export default function WeatherWidget({ location }) {
    // Mock data for demo
    const temp = 28;
    const humidity = 65;

    return (
        <Card variant="ceramic" className="flex items-center justify-between">
            <div>
                <div className="text-xs uppercase tracking-wider text-text-secondary font-medium mb-2">Weather</div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <LuCloudSun className="text-accent-ai" size={20} />
                        <span className="font-mono text-xl font-bold text-text-primary">{temp}°</span>
                    </div>
                    <div className="h-8 w-px bg-border-subtle"></div>
                    <div className="flex items-center gap-2">
                        <LuDroplets className="text-accent-medical" size={20} />
                        <span className="font-mono text-xl font-bold text-text-primary">{humidity}%</span>
                    </div>
                </div>
                <p className="text-xs text-text-secondary mt-2 font-medium">High humidity: Light hydration</p>
            </div>
        </Card>
    );
}
