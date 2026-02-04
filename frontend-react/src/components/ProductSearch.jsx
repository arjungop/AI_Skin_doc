import React, { useState } from 'react';
import { FaSearch, FaStar, FaLeaf } from 'react-icons/fa';

const ProductSearch = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        try {
            const res = await fetch('http://127.0.0.1:8000/recommendations/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, condition: "" }) // condition could be dynamic later
            });

            if (res.ok) {
                const data = await res.json();
                setResults(data);
            }
        } catch (err) {
            console.error("Search failed", err);
        } finally {
            setLoading(false);
        }
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
                    <div key={product.id} className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow flex gap-4 animate-fade-in">
                        <div className="w-16 h-16 bg-gray-50 rounded-lg flex items-center justify-center flex-shrink-0">
                            {product.image ? (
                                <img src={product.image} alt={product.name} className="w-full h-full object-contain rounded-lg" />
                            ) : (
                                <FaLeaf className="text-green-400 text-xl" />
                            )}
                        </div>
                        <div className="flex-1">
                            <div className="flex justify-between items-start">
                                <div>
                                    <h3 className="font-semibold text-gray-900">{product.name}</h3>
                                    <p className="text-sm text-gray-500 font-medium">{product.brand}</p>
                                </div>
                                <div className="flex items-center bg-green-50 px-2 py-1 rounded text-xs font-bold text-green-700">
                                    AI Match: {Math.round(product.score * 100)}%
                                </div>
                            </div>

                            <div className="mt-2 flex gap-2">
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                                    Skincare
                                </span>
                                {/* Fake tags for demo */}
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700">
                                    Repair
                                </span>
                            </div>
                        </div>
                    </div>
                ))}

                {results.length === 0 && !loading && (
                    <div className="text-center py-10 text-gray-400 text-sm">
                        No history. Try searching for "anti-aging cream"
                    </div>
                )}
            </div>
        </div>
    );
};

export default ProductSearch;
