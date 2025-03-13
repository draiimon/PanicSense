
import React from 'react';

export default function About() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <section>
        <h1 className="text-3xl font-bold text-slate-800 mb-4">About Disaster Sentiment Analysis</h1>
        <p className="text-slate-600 mb-6">
          The Disaster Sentiment Analysis System is an advanced platform designed to analyze and categorize emotional responses 
          during disaster events through social media and other text sources. By leveraging artificial intelligence and natural
          language processing techniques, our system helps emergency responders, government agencies, and disaster management
          teams understand public sentiment to better respond to crisis situations.
        </p>
      </section>

      <section className="bg-white p-6 rounded-lg shadow-sm">
        <h2 className="text-2xl font-bold text-slate-800 mb-4">Our Mission</h2>
        <p className="text-slate-600 mb-4">
          To provide accurate, real-time sentiment analysis of disaster-related communications to improve
          response coordination, resource allocation, and public support during crisis events.
        </p>
      </section>
      
      <section className="bg-white p-6 rounded-lg shadow-sm">
        <h2 className="text-2xl font-bold text-slate-800 mb-4">System Features</h2>
        <ul className="list-disc pl-6 text-slate-600 space-y-2">
          <li>Multi-language support with focus on English and Filipino/Tagalog</li>
          <li>Real-time analysis of social media posts</li>
          <li>CSV file batch processing for large datasets</li>
          <li>Sentiment categorization (Panic, Fear/Anxiety, Disbelief, Resilience, Neutral)</li>
          <li>Confidence scoring and detailed explanation of analysis</li>
          <li>Geographic and temporal visualization of sentiment trends</li>
          <li>Automatic disaster event detection and categorization</li>
        </ul>
      </section>
      
      <section className="bg-white p-6 rounded-lg shadow-sm">
        <h2 className="text-2xl font-bold text-slate-800 mb-4">Our Team</h2>
        <div className="grid md:grid-cols-3 gap-6 mt-4">
          <div className="text-center">
            <div className="w-32 h-32 bg-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
              <span className="text-4xl text-gray-500">MC</span>
            </div>
            <h3 className="font-bold text-slate-800">Mark Andrei R. Castillo</h3>
            <p className="text-slate-600">Lead Developer</p>
          </div>
          
          <div className="text-center">
            <div className="w-32 h-32 bg-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
              <span className="text-4xl text-gray-500">IG</span>
            </div>
            <h3 className="font-bold text-slate-800">Ivahnn Garcia</h3>
            <p className="text-slate-600">Data Scientist</p>
          </div>
          
          <div className="text-center">
            <div className="w-32 h-32 bg-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
              <span className="text-4xl text-gray-500">JG</span>
            </div>
            <h3 className="font-bold text-slate-800">Julia Daphne Ngan Gatdula</h3>
            <p className="text-slate-600">UX Designer</p>
          </div>
        </div>
      </section>
    </div>
  );
}
