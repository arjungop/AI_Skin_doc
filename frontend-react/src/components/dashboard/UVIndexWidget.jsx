import { useEffect, useState } from 'react';
import { LuSun, LuUmbrella } from 'react-icons/lu';
import Card from '../Card';

export default function UVIndexWidget({ location }) {
    const [uv, setUv] = useState(0);

    useEffect(() => {
        if (location) {
            // Mocking UV API for demo (OpenUV requires key). 
            // Logic: lat closer to 0 = higher UV.
            const mockUV = Math.floor(Math.random() * 8) + 1;
            setUv(mockUV);
        }
    }, [location]);

    const getRiskLevel = (val) => {
        if (val <= 2) return { text: 'Low', color: 'text-accent-medical' };
        if (val <= 5) return { text: 'Moderate', color: 'text-yellow-600' };
        if (val <= 7) return { text: 'High', color: 'text-orange-600' };
        return { text: 'Very High', color: 'text-red-600' };
    };

    const risk = getRiskLevel(uv);

    return (
        <Card variant="ceramic" className="flex items-center justify-between">
            <div>
                <div className="text-xs uppercase tracking-wider text-text-secondary font-medium mb-2">UV Index</div>
                <div className="flex items-baseline gap-3">
                    <span className="font-mono text-3xl font-bold text-text-primary">{uv}</span>
                    <span className={`text-xs font-mono px-2 py-1 rounded-full bg-surface ${risk.color} font-medium`}>
                        {risk.text}
                    </span>
                </div>
                <p className="text-xs text-text-secondary mt-2 font-medium">
                    {uv > 5 ? 'Wear SPF 50+' : 'No protection needed'}
                </p>
            </div>
            <div className={`h-12 w-12 rounded-full bg-surface flex items-center justify-center ${risk.color}`}>
                {uv > 5 ? <LuSun size={24} /> : <LuUmbrella size={24} />}
            </div>
        </Card>
    );
}
