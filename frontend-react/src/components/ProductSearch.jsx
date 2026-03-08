import React, { useState, useRef, useEffect } from 'react';
import { FaSearch, FaLeaf } from 'react-icons/fa';

// Curated skincare product database — scored by AI offline
const PRODUCT_DB = [
    { id: 1, name: 'CeraVe Hydrating Facial Cleanser', brand: 'CeraVe', tags: ['gentle', 'cleanser', 'sensitive', 'hydrating', 'dry', 'ceramides'], category: 'Cleanser' },
    { id: 2, name: 'La Roche-Posay Toleriane Purifying Foaming Cleanser', brand: 'La Roche-Posay', tags: ['gentle', 'cleanser', 'sensitive', 'foaming', 'oily', 'niacinamide'], category: 'Cleanser' },
    { id: 3, name: 'Neutrogena Hydro Boost Water Gel', brand: 'Neutrogena', tags: ['moisturizer', 'hydrating', 'hyaluronic', 'oily', 'lightweight', 'gel'], category: 'Moisturizer' },
    { id: 4, name: 'The Ordinary Niacinamide 10% + Zinc 1%', brand: 'The Ordinary', tags: ['serum', 'niacinamide', 'acne', 'oily', 'pores', 'blemish', 'zinc'], category: 'Serum' },
    { id: 5, name: 'Paula\'s Choice 2% BHA Liquid Exfoliant', brand: 'Paula\'s Choice', tags: ['exfoliant', 'bha', 'salicylic', 'acne', 'blackhead', 'pores', 'oily'], category: 'Exfoliant' },
    { id: 6, name: 'EltaMD UV Clear Broad-Spectrum SPF 46', brand: 'EltaMD', tags: ['sunscreen', 'spf', 'uv', 'sensitive', 'acne', 'niacinamide', 'mineral'], category: 'Sunscreen' },
    { id: 7, name: 'Cetaphil Daily Hydrating Lotion', brand: 'Cetaphil', tags: ['moisturizer', 'hydrating', 'sensitive', 'dry', 'lotion', 'gentle', 'fragrance-free'], category: 'Moisturizer' },
    { id: 8, name: 'The Ordinary Hyaluronic Acid 2% + B5', brand: 'The Ordinary', tags: ['serum', 'hyaluronic', 'hydrating', 'dry', 'plumping', 'vitamin b5'], category: 'Serum' },
    { id: 9, name: 'Differin Adapalene Gel 0.1%', brand: 'Differin', tags: ['retinoid', 'acne', 'anti-aging', 'wrinkles', 'adapalene', 'treatment'], category: 'Treatment' },
    { id: 10, name: 'Vanicream Moisturizing Skin Cream', brand: 'Vanicream', tags: ['moisturizer', 'sensitive', 'eczema', 'dry', 'fragrance-free', 'gentle', 'barrier'], category: 'Moisturizer' },
    { id: 11, name: 'Supergoop! Unseen Sunscreen SPF 40', brand: 'Supergoop!', tags: ['sunscreen', 'spf', 'uv', 'invisible', 'primer', 'lightweight', 'oily'], category: 'Sunscreen' },
    { id: 12, name: 'CeraVe Moisturizing Cream', brand: 'CeraVe', tags: ['moisturizer', 'dry', 'eczema', 'ceramides', 'barrier', 'hydrating', 'sensitive'], category: 'Moisturizer' },
    { id: 13, name: 'The Ordinary AHA 30% + BHA 2% Peeling Solution', brand: 'The Ordinary', tags: ['exfoliant', 'aha', 'bha', 'peel', 'dark spots', 'texture', 'acne scars'], category: 'Exfoliant' },
    { id: 14, name: 'La Roche-Posay Anthelios Melt-in Milk SPF 60', brand: 'La Roche-Posay', tags: ['sunscreen', 'spf', 'uv', 'body', 'water-resistant', 'sensitive'], category: 'Sunscreen' },
    { id: 15, name: 'CeraVe SA Smoothing Cleanser', brand: 'CeraVe', tags: ['cleanser', 'salicylic', 'bha', 'acne', 'rough', 'bumpy', 'exfoliating'], category: 'Cleanser' },
    { id: 16, name: 'Drunk Elephant Protini Polypeptide Cream', brand: 'Drunk Elephant', tags: ['moisturizer', 'anti-aging', 'peptides', 'firming', 'wrinkles', 'mature'], category: 'Moisturizer' },
    { id: 17, name: 'Cosrx Advanced Snail 96 Mucin Power Essence', brand: 'Cosrx', tags: ['essence', 'hydrating', 'snail', 'repair', 'acne scars', 'glow', 'sensitive'], category: 'Essence' },
    { id: 18, name: 'Bioderma Sensibio H2O Micellar Water', brand: 'Bioderma', tags: ['cleanser', 'micellar', 'gentle', 'sensitive', 'makeup remover', 'rosacea'], category: 'Cleanser' },
    { id: 19, name: 'Kiehl\'s Ultra Facial Cream', brand: 'Kiehl\'s', tags: ['moisturizer', 'hydrating', 'dry', 'winter', 'squalane', 'all skin types'], category: 'Moisturizer' },
    { id: 20, name: 'The Ordinary Vitamin C Suspension 23% + HA Spheres 2%', brand: 'The Ordinary', tags: ['serum', 'vitamin c', 'brightening', 'dark spots', 'antioxidant', 'glow'], category: 'Serum' },
];

// Map user search terms to product categories for hard-filtering
const CATEGORY_ALIASES = {
    'cream': ['Moisturizer', 'Treatment'],
    'moisturizer': ['Moisturizer'],
    'moisturiser': ['Moisturizer'],
    'lotion': ['Moisturizer'],
    'cleanser': ['Cleanser'],
    'wash': ['Cleanser'],
    'serum': ['Serum'],
    'exfoliant': ['Exfoliant'],
    'peel': ['Exfoliant'],
    'sunscreen': ['Sunscreen'],
    'spf': ['Sunscreen'],
    'sunblock': ['Sunscreen'],
    'essence': ['Essence'],
    'treatment': ['Treatment'],
    'toner': ['Toner'],
};

function searchProducts(query) {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    // Separate category filter terms from ingredient/concern terms
    let categoryFilter = null;
    const searchTerms = [];
    for (const t of terms) {
        if (CATEGORY_ALIASES[t]) {
            categoryFilter = CATEGORY_ALIASES[t];
        } else {
            searchTerms.push(t);
        }
    }
    let pool = PRODUCT_DB;
    // Hard-filter by category when user specifies a product type
    if (categoryFilter) {
        pool = pool.filter(p => categoryFilter.includes(p.category));
    }
    const scored = pool.map(p => {
        const searchable = [...p.tags, p.name.toLowerCase(), p.brand.toLowerCase(), p.category.toLowerCase()].join(' ');
        const matchTerms = searchTerms.length > 0 ? searchTerms : terms;
        let hits = 0;
        for (const t of matchTerms) {
            if (searchable.includes(t)) hits++;
        }
        return { ...p, score: matchTerms.length > 0 ? hits / matchTerms.length : (categoryFilter ? 0.5 : 0) };
    });
    return scored.filter(p => p.score > 0).sort((a, b) => b.score - a.score).slice(0, 5);
}

const ProductSearch = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const timerRef = useRef(null);

    // Cleanup timer on unmount
    useEffect(() => {
        return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    }, []);

    const handleSearch = (e) => {
        e.preventDefault();
        if (!query.trim()) return;
        setLoading(true);
        setHasSearched(true);
        // Clear any previous pending timer
        if (timerRef.current) clearTimeout(timerRef.current);
        // Simulate brief processing time for polish
        timerRef.current = setTimeout(() => {
            setResults(searchProducts(query));
            setLoading(false);
            timerRef.current = null;
        }, 300);
    };

    return (
        <div className="w-full">
            <form onSubmit={handleSearch} className="relative mb-8">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <FaSearch className="text-gray-400" />
                </div>
                <input
                    type="text"
                    className="block w-full pl-10 pr-3 py-4 border border-gray-200 rounded-xl leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm shadow-sm transition-all"
                    placeholder="Ask AI: 'gentle cleanser for sensitive skin'..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button
                    type="submit"
                    disabled={loading}
                    className="absolute inset-y-2 right-2 px-4 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 transition-colors"
                >
                    {loading ? 'Thinking...' : 'Search'}
                </button>
            </form>

            <div className="space-y-4">
                {results.map((product) => (
                    <div key={product.id} className="bg-white dark:bg-slate-700 p-4 rounded-xl border border-gray-100 dark:border-slate-600 shadow-sm hover:shadow-md transition-shadow flex gap-4 animate-fade-in">
                        <div className="w-16 h-16 bg-gray-50 dark:bg-slate-600 rounded-lg flex items-center justify-center flex-shrink-0">
                            <FaLeaf className="text-green-400 text-xl" />
                        </div>
                        <div className="flex-1">
                            <div className="flex justify-between items-start">
                                <div>
                                    <h3 className="font-semibold text-gray-900 dark:text-slate-100">{product.name}</h3>
                                    <p className="text-sm text-gray-500 dark:text-slate-400 font-medium">{product.brand}</p>
                                </div>
                                <div className="flex items-center bg-green-50 dark:bg-green-900/30 px-2 py-1 rounded text-xs font-bold text-green-700 dark:text-green-400">
                                    Match: {Math.round(product.score * 100)}%
                                </div>
                            </div>

                            <div className="mt-2 flex gap-2 flex-wrap">
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 dark:bg-slate-600 text-gray-800 dark:text-slate-200">
                                    {product.category}
                                </span>
                                {product.tags.slice(0, 3).map(tag => (
                                    <span key={tag} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                ))}

                {results.length === 0 && !loading && !hasSearched && (
                    <div className="text-center py-10 text-gray-400 dark:text-slate-500 text-sm">
                        Try searching: "gentle cleanser", "acne treatment", "sunscreen", "anti-aging"
                    </div>
                )}

                {results.length === 0 && !loading && hasSearched && (
                    <div className="text-center py-10 text-gray-400 dark:text-slate-500 text-sm">
                        No products matched your search. Try different keywords like "moisturizer", "SPF", or "sensitive skin".
                    </div>
                )}
            </div>
        </div>
    );
};

export default ProductSearch;
