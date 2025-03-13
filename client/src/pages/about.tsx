
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function About() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">About PanicSense PH</h1>
      
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Our Mission</CardTitle>
          <CardDescription>Empowering communities through disaster sentiment analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            PanicSense PH is dedicated to providing real-time disaster sentiment analysis to help communities
            and emergency responders better understand public reactions during crisis situations. By analyzing
            social media posts and other text sources, we identify patterns of panic, fear, resilience, and
            other emotions to guide more effective disaster response.
          </p>
        </CardContent>
      </Card>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Our Team</CardTitle>
            <CardDescription>The founders behind PanicSense PH</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold">John Doe</h3>
                <p className="text-sm text-slate-500">Co-Founder & CEO</p>
                <p className="mt-1">Disaster management specialist with 10+ years of experience in emergency response systems.</p>
              </div>
              <div>
                <h3 className="font-semibold">Jane Smith</h3>
                <p className="text-sm text-slate-500">Co-Founder & CTO</p>
                <p className="mt-1">AI researcher specializing in natural language processing and sentiment analysis.</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Our Technology</CardTitle>
            <CardDescription>Cutting-edge sentiment analysis for disaster response</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              PanicSense PH leverages advanced AI and machine learning algorithms to analyze text data in real-time.
              Our system is capable of processing both English and Filipino content, recognizing cultural nuances
              and local expressions to accurately assess sentiment during disaster situations.
            </p>
            <p>
              The platform categorizes sentiments into Panic, Fear/Anxiety, Disbelief, Resilience, and Neutral,
              providing actionable insights for emergency responders and community leaders.
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Contact Us</CardTitle>
          <CardDescription>Get in touch with the PanicSense PH team</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Email: admin@panicsense.ph</p>
          <p>Address: Manila, Philippines</p>
        </CardContent>
      </Card>
    </div>
  );
}
