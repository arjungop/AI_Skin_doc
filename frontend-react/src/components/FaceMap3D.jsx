import React, { useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, useGLTF, Html } from '@react-three/drei'
import * as THREE from 'three'

function Face(props) {
    const meshRef = useRef()
    const [hoveredZone, setHoveredZone] = useState(null)

    // In a real app, we would load a detailed GLTF model here
    // const { nodes, materials } = useGLTF('/models/head.gltf')

    // For this demo, we construct a geometric abstraction of a face
    // Using a group of shapes to represent zones

    return (
        <group {...props} dispose={null}>
            {/* Base Head Shape */}
            <mesh position={[0, 0, 0]} scale={[1, 1.3, 1]}>
                <sphereGeometry args={[1, 32, 32]} />
                <meshStandardMaterial color="#fce4d4" roughness={0.5} />
            </mesh>

            {/* Forehead Zone */}
            <Zone
                position={[0, 0.7, 0.85]}
                scale={[0.8, 0.5, 0.2]}
                label="Forehead"
                onSelect={props.onSelect}
                selected={props.selectedZones.includes('Forehead')}
            />

            {/* Nose Zone */}
            <Zone
                position={[0, 0, 1.05]}
                scale={[0.25, 0.6, 0.2]}
                label="Nose"
                onSelect={props.onSelect}
                selected={props.selectedZones.includes('Nose')}
            />

            {/* Left Cheek */}
            <Zone
                position={[-0.6, -0.2, 0.8]}
                scale={[0.5, 0.6, 0.3]}
                label="Left Cheek"
                onSelect={props.onSelect}
                selected={props.selectedZones.includes('Left Cheek')}
            />

            {/* Right Cheek */}
            <Zone
                position={[0.6, -0.2, 0.8]}
                scale={[0.5, 0.6, 0.3]}
                label="Right Cheek"
                onSelect={props.onSelect}
                selected={props.selectedZones.includes('Right Cheek')}
            />

            {/* Chin */}
            <Zone
                position={[0, -0.9, 0.9]}
                scale={[0.5, 0.4, 0.2]}
                label="Chin"
                onSelect={props.onSelect}
                selected={props.selectedZones.includes('Chin')}
            />

            {/* Eyes (Visual Only) */}
            <mesh position={[-0.35, 0.2, 0.9]}>
                <sphereGeometry args={[0.1, 16, 16]} />
                <meshStandardMaterial color="#333" />
            </mesh>
            <mesh position={[0.35, 0.2, 0.9]}>
                <sphereGeometry args={[0.1, 16, 16]} />
                <meshStandardMaterial color="#333" />
            </mesh>

        </group>
    )
}

function Zone({ position, scale, label, onSelect, selected }) {
    const [hovered, setHover] = useState(false)

    useFrame((state) => {
        // Subtle pulsing animation if selected
        if (selected) {
            // ref.current.scale.set(...) 
        }
    })

    return (
        <mesh
            position={position}
            scale={scale}
            onClick={(e) => { e.stopPropagation(); onSelect(label) }}
            onPointerOver={(e) => { e.stopPropagation(); setHover(true); document.body.style.cursor = 'pointer' }}
            onPointerOut={(e) => { setHover(false); document.body.style.cursor = 'default' }}
            renderOrder={1} // Render on top of base
        >
            <boxGeometry args={[1, 1, 1]} />
            <meshPhysicalMaterial
                color={selected ? '#D4AF37' : (hovered ? '#e67e22' : '#fce4d4')}
                transparent={true}
                opacity={selected || hovered ? 0.6 : 0.0}
                roughness={0}
                transmission={0.5}
                thickness={1}
            />

            {(hovered || selected) && (
                <Html distanceFactor={10}>
                    <div className="bg-slate-900/80 text-white px-2 py-1 rounded text-xs whitespace-nowrap backdrop-blur-md border border-white/20">
                        {label}
                        {selected && <span className="ml-1 text-[#D4AF37]">✓</span>}
                    </div>
                </Html>
            )}
        </mesh>
    )
}

export default function FaceMap3D({ onZoneSelect, selectedZones = [] }) {
    return (
        <div className="w-full h-full min-h-[300px] relative rounded-xl overflow-hidden bg-gradient-to-b from-slate-50/50 to-white/50 border border-slate-100/50">
            <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
                <ambientLight intensity={0.7} />
                <pointLight position={[10, 10, 10]} intensity={1} color="#fff" />
                <pointLight position={[-10, -10, -10]} intensity={0.5} color="#D4AF37" />

                <Face onSelect={onZoneSelect} selectedZones={selectedZones} />

                <OrbitControls
                    enableZoom={false}
                    minPolarAngle={Math.PI / 4}
                    maxPolarAngle={Math.PI / 1.5}
                    rotateSpeed={0.5}
                />
            </Canvas>
            <div className="absolute bottom-2 left-0 right-0 text-center pointer-events-none">
                <p className="text-[10px] text-slate-400 font-medium uppercase tracking-widest">
                    Interactive 3D Model
                </p>
            </div>
        </div>
    )
}
