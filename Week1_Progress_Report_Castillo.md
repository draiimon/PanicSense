Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 1 / January 20 - January 24, 2025



Initial Architecture Design for PanicSensePH Platform



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on establishing the foundational architecture for our PanicSensePH disaster monitoring platform. I successfully designed the initial system architecture diagram showing data flow between frontend React components, Express backend services, and PostgreSQL database. This architecture emphasizes real-time data processing capabilities needed for disaster response.

I conducted research on state-of-the-art sentiment analysis approaches, particularly focusing on LSTM and MBERT applications in disaster management contexts. Based on this research, I drafted preliminary specifications for our sentiment analysis pipeline that will form the core of our platform.

Additionally, I implemented the initial database schema using Drizzle ORM, defining all essential tables for storing sentiment posts, disaster events, and user data. This schema design ensures proper normalization while maintaining compatibility with our planned sentiment analysis components.



Techniques, Tools, and Methodologies Used

I utilized PostgreSQL with Drizzle ORM for database design, implementing normalized schemas for disaster-related data. TypeScript provided type safety throughout the system, particularly for data model definitions. For backend implementation, I used Express.js to establish RESTful API endpoints for data exchange.

For frontend development, I implemented React with Vite for an efficient development environment with hot module replacement. The development workflow was structured using Git Flow methodology to facilitate parallel development across team members.



Reflection: Problems Encountered and Lessons Learned

The most significant challenge was designing a schema that could accommodate both structured data from official disaster reports and unstructured content from social media sources. I learned that planning for future expansion from the beginning is crucial, particularly regarding the sentiment analysis models we'll implement.

I also encountered difficulties determining the optimal approach for multilingual support, as our system needs to handle both English and Filipino text, often with code-switching. Through research, I identified MBERT as a promising solution for this challenge, which will be implemented in upcoming sprints.