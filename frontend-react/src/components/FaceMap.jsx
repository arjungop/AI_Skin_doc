import React from 'react'

export default function FaceMap({ onZoneSelect, selectedZones = [] }) {
    const isSelected = (zone) => selectedZones.includes(zone)

    const zones = [
        { id: 'forehead', d: 'M100,60 Q150,40 200,60 L200,90 Q150,110 100,90 Z', label: 'Forehead' },
        { id: 'mid_brows', d: 'M135,90 L165,90 L165,110 L135,110 Z', label: 'Mid Brows' },
        { id: 'nose', d: 'M135,110 L165,110 L170,160 L130,160 Z', label: 'Nose' },
        { id: 'left_cheek', d: 'M60,100 Q90,100 130,120 L130,180 Q80,200 50,150 Z', label: 'Left Cheek' },
        { id: 'right_cheek', d: 'M240,100 Q210,100 170,120 L170,180 Q220,200 250,150 Z', label: 'Right Cheek' },
        { id: 'chin', d: 'M120,200 Q150,220 180,200 L170,180 L130,180 Z', label: 'Chin' },
    ]

    // Simple abstract face shape path for background
    const faceOutline = "M50,100 Q50,50 150,50 Q250,50 250,100 Q250,220 150,250 Q50,220 50,100 Z"

    return (
        <div className="relative w-full max-w-[300px] mx-auto aspect-[3/4]">
            <svg viewBox="0 0 300 300" className="w-full h-full drop-shadow-xl">
                {/* Face Outline / Base Skin */}
                <path d={faceOutline} fill="#fce4d4" stroke="#eecbb5" strokeWidth="4" />

                {/* Zones */}
                {zones.map((zone) => (
                    <g key={zone.id} onClick={() => onZoneSelect(zone.label)} className="cursor-pointer group">
                        <path
                            d={zone.d}
                            fill={isSelected(zone.label) ? '#fbbf24' : 'rgba(255,255,255,0.3)'}
                            stroke={isSelected(zone.label) ? '#d97706' : 'rgba(0,0,0,0.05)'}
                            strokeWidth="2"
                            className="transition-colors duration-300 group-hover:fill-amber-200/50"
                        />
                        {/* Label on Hover */}
                        <title>{zone.label}</title>
                    </g>
                ))}

                {/* Eyes (Visual only) */}
                <ellipse cx="100" cy="120" rx="15" ry="8" fill="#fff" opacity="0.8" />
                <circle cx="100" cy="120" r="5" fill="#555" />
                <ellipse cx="200" cy="120" rx="15" ry="8" fill="#fff" opacity="0.8" />
                <circle cx="200" cy="120" r="5" fill="#555" />

                {/* Mouth (Visual only) */}
                <path d="M110,190 Q150,210 190,190" fill="none" stroke="#d99f8c" strokeWidth="3" strokeLinecap="round" />

            </svg>
        </div>
    )
}
