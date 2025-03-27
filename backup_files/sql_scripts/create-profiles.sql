
-- First clear existing data
TRUNCATE TABLE profile_images;

-- Insert updated profile data
INSERT INTO profile_images (name, role, image_url, description) VALUES
('Mark Andrei R. Castillo', 'Core System Architecture & Machine Learning', '/assets/drei.jpg', 'Leads the development of our advanced ML pipelines and system architecture'),
('Ivahnn B. Garcia', 'Frontend Development & User Experience', '/assets/van.jpg', 'Creates intuitive and responsive user interfaces for seamless interaction'),
('Julia Daphne Ngan-Gatdula', 'Data Resources & Information Engineering', '/assets/julia.jpg', 'Manages data infrastructure and information processing systems');
