const jobPostings = [
    {
     jobId: 1,
      jobTitle: "MERN Stack Developer",
      company: "Tech Innovators inc",
      location: "San Francisco, CA (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "Looking for a MERN Stack Developer to build dynamic web applications.",
      jobResponsibilities: [
        "Develop front-end using React.js.",
        "Design and manage database with MongoDB.",
        "Implement server-side logic with Node.js and Express.js."
      ],
      preferredQualifications: [
        "Experience with RESTful APIs.",
        "Knowledge of cloud services (AWS/GCP).",
        "Strong understanding of CI/CD pipelines."
      ],
      applicationDeadline: "2024-08-01"
    },
    {
     jobId: 2,
      jobTitle: "Full Stack Developer 1",
      company: "Innovative Solutions",
      location: "New York, NY (Remote)",
      jobType: "Full-Time",
      salary: "$100,000 - $130,000",
      jobDescription: "We need a Full Stack Developer experienced in both front-end and back-end development.",
      jobResponsibilities: [
        "Develop and maintain web applications.",
        "Collaborate with designers and other developers.",
        "Ensure responsive design and cross-browser compatibility."
      ],
      preferredQualifications: [
        "Experience with JavaScript frameworks.",
        "Knowledge of database management (SQL/NoSQL).",
        "Familiarity with version control systems (Git)."
      ],
      applicationDeadline: "2024-07-25"
    },
    {
     jobId: 3,
      jobTitle: "Python Developer Scientist",
      company: "Data Wizards",
      location: "Boston, MA (Remote)",
      jobType: "Full-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "We are looking for a proficient Python Developer to create and sustain our software solutions.",
      jobResponsibilities: [
        "Write reusable, testable, and efficient code.",
        "Design and implement low-latency, high-availability applications.",
        "Integrate user-facing elements with server-side logic."
      ],
      preferredQualifications: [
        "Familiarity with front-end technologies (like JavaScript and HTML5).",
        "Understanding of accessibility and security compliance.",
        "Knowledge of user authentication and authorization."
      ],
      applicationDeadline: "2024-07-20"
    },
    {
     jobId: 4,
      jobTitle: "Data Scientist-1",
      company: "Data Insights",
      location: "Chicago, IL (Remote)",
      jobType: "Full-Time",
      salary: "$95,000 - $125,000",
      jobDescription: "We are looking for a Data Scientist to analyze large datasets and derive actionable insights.",
      jobResponsibilities: [
        "Develop and implement data models.",
        "Analyze data trends and patterns.",
        "Collaborate with cross-functional teams."
      ],
      preferredQualifications: [
        "Experience with machine learning frameworks.",
        "Proficiency in Python or R.",
        "Strong statistical analysis skills."
      ],
      applicationDeadline: "2024-07-30"
    },
    {
     jobId: 5,
      jobTitle: "Project Manager",
      company: "Business Solutions",
      location: "Seattle, WA (Remote)",
      jobType: "Full-Time",
      salary: "$80,000 - $100,000",
      jobDescription: "Seeking an experienced Project Manager to oversee various projects and ensure timely delivery.",
      jobResponsibilities: [
        "Develop project plans and timelines.",
        "Coordinate with stakeholders.",
        "Monitor project progress and adjust as necessary."
      ],
      preferredQualifications: [
        "Experience with project management software (e.g., MS Project).",
        "Strong leadership and communication skills.",
        "PMP certification is a plus."
      ],
      applicationDeadline: "2024-08-10"
    },
    {
     jobId: 6,
      jobTitle: "UX/UI Designer",
      company: "Creative Minds",
      location: "Austin, TX (Remote)",
      jobType: "Full-Time",
      salary: "$75,000 - $95,000",
      jobDescription: "We need a UX/UI Designer to create intuitive and visually appealing user interfaces.",
      jobResponsibilities: [
        "Design user interfaces for web and mobile applications.",
        "Conduct user research and usability testing.",
        "Collaborate with product and development teams."
      ],
      preferredQualifications: [
        "Proficiency in design tools (Sketch, Adobe XD).",
        "Strong portfolio showcasing UX/UI design work.",
        "Experience with wireframing and prototyping."
      ],
      applicationDeadline: "2024-08-05"
    },
    {
     jobId: 7,
      jobTitle: "Systems Analyst",
      company: "Tech Innovators",
      location: "Phoenix, AZ (Remote)",
      jobType: "Full-Time",
      salary: "$70,000 - $90,000",
      jobDescription: "We need a Systems Analyst to implement, maintain, and support IT systems.",
      jobResponsibilities: [
        "Analyze and design IT solutions.",
        "Document system requirements and specifications.",
        "Collaborate with developers and end-users."
      ],
      preferredQualifications: [
        "Experience with SQL and software development.",
        "Strong analytical and problem-solving skills.",
        "Knowledge of business analysis techniques."
      ],
      applicationDeadline: "2024-08-15"
    },
    {
     jobId: 8,
      jobTitle: "Digital Marketing Specialist",
      company: "Marketing Pros",
      location: "Los Angeles, CA (Remote)",
      jobType: "Full-Time",
      salary: "$65,000 - $85,000",
      jobDescription: "Looking for a Digital Marketing Specialist to develop and execute marketing campaigns.",
      jobResponsibilities: [
        "Create and manage digital marketing campaigns.",
        "Analyze and report on campaign performance.",
        "Collaborate with the marketing team."
      ],
      preferredQualifications: [
        "Experience with SEO and SEM.",
        "Proficiency in Google Analytics and AdWords.",
        "Strong understanding of social media marketing."
      ],
      applicationDeadline: "2024-08-20"
    },
    {
     jobId: 9,
      jobTitle: "Network Administrator",
      company: "IT Solutions",
      location: "Miami, FL (On-site)",
      jobType: "Full-Time",
      salary: "$60,000 - $80,000",
      jobDescription: "Seeking a Network Administrator to maintain and optimize our network infrastructure.",
      jobResponsibilities: [
        "Install and configure network hardware and software.",
        "Monitor network performance and troubleshoot issues.",
        "Ensure network security and compliance."
      ],
      preferredQualifications: [
        "Experience with networking protocols and services.",
        "Knowledge of firewalls and security systems.",
        "Certification (CCNA, CCNP) is a plus."
      ],
      applicationDeadline: "2024-08-25"
    },
    {
     jobId: 10,
      jobTitle: "Financial Analyst",
      company: "Finance Experts",
      location: "Houston, TX (Remote)",
      jobType: "Full-Time",
      salary: "$70,000 - $95,000",
      jobDescription: "Looking for a Financial Analyst to provide financial planning and analysis.",
      jobResponsibilities: [
        "Analyze financial data and create financial models.",
        "Prepare reports and forecasts.",
        "Assist in budgeting and financial planning."
      ],
      preferredQualifications: [
        "Experience with financial analysis and modeling.",
        "Proficiency in Excel and financial software.",
        "Strong analytical and problem-solving skills."
      ],
      applicationDeadline: "2024-09-01"
    },
    {
     jobId: 11,
      jobTitle: "Software Engineer",
      company: "Tech Solutions",
      location: "New York, NY (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "We are looking for a Software Engineer to develop and maintain software applications.",
      jobResponsibilities: [
        "Write clean and efficient code.",
        "Collaborate with cross-functional teams.",
        "Troubleshoot and debug applications."
      ],
      preferredQualifications: [
        "Experience with software development lifecycle.",
        "Proficiency in programming languages (e.g., Java, C++).",
        "Strong problem-solving skills."
      ],
      applicationDeadline: "2024-09-05"
    },
    {
     jobId: 12,
      jobTitle: "Human Resources Manager",
      company: "HR Pros",
      location: "Boston, MA (Remote)",
      jobType: "Full-Time",
      salary: "$80,000 - $100,000",
      jobDescription: "Seeking an HR Manager to oversee HR policies and procedures.",
      jobResponsibilities: [
        "Develop and implement HR strategies.",
        "Manage recruitment and selection processes.",
        "Oversee employee relations and performance management."
      ],
      preferredQualifications: [
        "Experience in HR management.",
        "Strong knowledge of labor laws and regulations.",
        "Excellent communication and interpersonal skills."
      ],
      applicationDeadline: "2024-09-10"
    },
    {
     jobId: 13,
      jobTitle: "Graphic Designer",
      company: "Creative Studio",
      location: "San Diego, CA (Remote)",
      jobType: "Full-Time",
      salary: "$65,000 - $85,000",
      jobDescription: "We need a Graphic Designer to create visually appealing graphics and layouts.",
      jobResponsibilities: [
        "Design graphics for digital and print media.",
        "Collaborate with marketing and design teams.",
        "Ensure brand consistency in all designs."
      ],
      preferredQualifications: [
        "Proficiency in Adobe Creative Suite.",
        "Strong portfolio showcasing design work.",
        "Experience with typography and color theory."
      ],
      applicationDeadline: "2024-09-15"
    },
    {
     jobId: 14,
      jobTitle: "Cybersecurity Analyst",
      company: "SecureTech",
      location: "Washington, D.C. (Remote)",
      jobType: "Full-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "Looking for a Cybersecurity Analyst to monitor and protect our systems.",
      jobResponsibilities: [
        "Monitor network for security breaches.",
        "Investigate security incidents.",
        "Implement security measures and protocols."
      ],
      preferredQualifications: [
        "Experience with SIEM tools.",
        "Knowledge of cybersecurity frameworks.",
        "Strong analytical and problem-solving skills."
      ],
      applicationDeadline: "2024-09-20"
    },
    {
     jobId: 15,
      jobTitle: "Cloud Solutions Architect",
      company: "Cloud Experts",
      location: "Seattle, WA (Remote)",
      jobType: "Full-Time",
      salary: "$100,000 - $130,000",
      jobDescription: "Seeking a Cloud Solutions Architect to design and implement cloud solutions.",
      jobResponsibilities: [
        "Develop cloud strategies and roadmaps.",
        "Design cloud architectures.",
        "Ensure cloud solutions are secure and scalable."
      ],
      preferredQualifications: [
        "Experience with AWS, Azure, or GCP.",
        "Knowledge of cloud security and compliance.",
        "Strong understanding of cloud computing concepts."
      ],
      applicationDeadline: "2024-09-25"
    },
    {
     jobId: 16,
      jobTitle: "Product Manager",
      company: "Innovative Products",
      location: "Boston, MA (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "Looking for a Product Manager to oversee the development and delivery of our products.",
      jobResponsibilities: [
        "Define product vision and strategy.",
        "Manage product lifecycle.",
        "Collaborate with cross-functional teams."
      ],
      preferredQualifications: [
        "Experience in product management.",
        "Strong understanding of Agile methodologies.",
        "Excellent communication and leadership skills."
      ],
      applicationDeadline: "2024-09-30"
    },
    {
     jobId: 17,
      jobTitle: "Content Writer",
      company: "Content Creators",
      location: "Los Angeles, CA (Remote)",
      jobType: "Full-Time",
      salary: "$55,000 - $75,000",
      jobDescription: "We need a Content Writer to create engaging and informative content.",
      jobResponsibilities: [
        "Write articles, blog posts, and social media content.",
        "Research and develop content ideas.",
        "Collaborate with marketing and design teams."
      ],
      preferredQualifications: [
        "Strong writing and editing skills.",
        "Experience with SEO best practices.",
        "Proficiency in content management systems."
      ],
      applicationDeadline: "2024-10-05"
    },
    {
     jobId: 18,
      jobTitle: "Sales Executive",
      company: "Sales Pros",
      location: "Houston, TX (Remote)",
      jobType: "Full-Time",
      salary: "$70,000 - $90,000",
      jobDescription: "Looking for a Sales Executive to drive sales and revenue growth.",
      jobResponsibilities: [
        "Identify and develop new business opportunities.",
        "Manage and grow client relationships.",
        "Achieve sales targets and goals."
      ],
      preferredQualifications: [
        "Experience in sales and business development.",
        "Strong negotiation and communication skills.",
        "Proficiency in CRM software."
      ],
      applicationDeadline: "2024-10-10"
    },
    {
     jobId: 19,
      jobTitle: "Business Intelligence Analyst",
      company: "BizTech",
      location: "Houston, TX (Remote)",
      jobType: "Full-Time",
      salary: "$80,000 - $105,000",
      jobDescription: "We need a Business Intelligence Analyst to assess business processes and provide data-driven insights.",
      jobResponsibilities: [
        "Develop and maintain BI solutions.",
        "Analyze data and generate reports.",
        "Collaborate with business stakeholders."
      ],
      preferredQualifications: [
        "Experience with BI tools (e.g., Tableau, Power BI).",
        "Strong analytical and problem-solving skills.",
        "Proficiency in SQL and data analysis."
      ],
      applicationDeadline: "2024-10-15"
    },
    {
     jobId: 20,
      jobTitle: "Customer Support Specialist",
      company: "HelpDesk Heroes",
      location: "Miami, FL (On-site)",
      jobType: "Full-Time",
      salary: "$50,000 - $65,000",
      jobDescription: "Seeking a Customer Support Specialist to provide technical assistance to computer users.",
      jobResponsibilities: [
        "Respond to customer inquiries and issues.",
        "Troubleshoot and resolve technical problems.",
        "Document support interactions and solutions."
      ],
      preferredQualifications: [
        "Experience in customer support or technical support.",
        "Strong communication and problem-solving skills.",
        "Knowledge of computer hardware and software."
      ],
      applicationDeadline: "2024-10-20"
    },
    {
     jobId: 21,
      jobTitle: "Operations Manager",
      company: "Business Solutions",
      location: "Seattle, WA (Remote)",
      jobType: "Full-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "We need an Operations Manager to oversee daily operations and ensure efficiency.",
      jobResponsibilities: [
        "Manage operational processes and workflows.",
        "Develop and implement operational strategies.",
        "Monitor and improve performance metrics."
      ],
      preferredQualifications: [
        "Experience in operations management.",
        "Strong organizational and leadership skills.",
        "Knowledge of process improvement methodologies."
      ],
      applicationDeadline: "2024-10-25"
    },
    {
     jobId: 22,
      jobTitle: "SEO Specialist",
      company: "Marketing Pros",
      location: "Los Angeles, CA (Remote)",
      jobType: "Full-Time",
      salary: "$60,000 - $80,000",
      jobDescription: "Looking for an SEO Specialist to improve our website's search engine ranking.",
      jobResponsibilities: [
        "Conduct keyword research and analysis.",
        "Optimize website content and structure.",
        "Monitor and report on SEO performance."
      ],
      preferredQualifications: [
        "Experience with SEO tools (e.g., SEMrush, Ahrefs).",
        "Strong understanding of search engine algorithms.",
        "Proficiency in HTML and web analytics."
      ],
      applicationDeadline: "2024-10-30"
    },
    {
     jobId: 23,
      jobTitle: "Mobile App Developer",
      company: "App Masters",
      location: "New York, NY (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $115,000",
      jobDescription: "Seeking a Mobile App Developer to create high-quality mobile applications.",
      jobResponsibilities: [
        "Develop mobile apps for Android and iOS.",
        "Collaborate with designers and developers.",
        "Ensure app performance and stability."
      ],
      preferredQualifications: [
        "Proficiency in Swift and Kotlin.",
        "Experience with React Native.",
        "Strong understanding of mobile development frameworks."
      ],
      applicationDeadline: "2024-11-05"
    },
    {
     jobId: 24,
      jobTitle: "MERN Stack Developer",
      company: "Tech Innovators",
      location: "San Francisco, CA (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "Looking for a MERN Stack Developer to build dynamic web applications.",
      jobResponsibilities: [
        "Develop front-end using React.js.",
        "Design and manage database with MongoDB.",
        "Implement server-side logic with Node.js and Express.js."
      ],
      preferredQualifications: [
        "Experience with RESTful APIs.",
        "Knowledge of cloud services (AWS/GCP).",
        "Strong understanding of CI/CD pipelines."
      ],
      applicationDeadline: "2024-11-10"
    },
    {
     jobId: 25,
      jobTitle: "Full Stack Developer 2",
      company: "Innovative Solutions",
      location: "New York, NY (Remote)",
      jobType: "Full-Time",
      salary: "$100,000 - $130,000",
      jobDescription: "We need a Full Stack Developer experienced in both front-end and back-end development.",
      jobResponsibilities: [
        "Develop and maintain web applications.",
        "Collaborate with designers and other developers.",
        "Ensure responsive design and cross-browser compatibility."
      ],
      preferredQualifications: [
        "Experience with JavaScript frameworks.",
        "Knowledge of database management (SQL/NoSQL).",
        "Familiarity with version control systems (Git)."
      ],
      applicationDeadline: "2024-11-15"
    },
    {
     jobId: 26,
      jobTitle: "Python Developer",
      company: "Data Wizards",
      location: "Boston, MA (Remote)",
      jobType: "Full-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "Seeking a skilled Python Developer to develop and maintain our software solutions.",
      jobResponsibilities: [
        "Write reusable, testable, and efficient code.",
        "Design and implement low-latency, high-availability applications.",
        "Integrate user-facing elements with server-side logic."
      ],
      preferredQualifications: [
        "Familiarity with front-end technologies (like JavaScript and HTML5).",
        "Understanding of accessibility and security compliance.",
        "Knowledge of user authentication and authorization."
      ],
      applicationDeadline: "2024-11-20"
    },
    {
     jobId: 27,
      jobTitle: "Data Scientist-2",
      company: "Data Insights",
      location: "Chicago, IL (Remote)",
      jobType: "Full-Time",
      salary: "$95,000 - $125,000",
      jobDescription: "We are looking for a Data Scientist to analyze large datasets and derive actionable insights.",
      jobResponsibilities: [
        "Develop and implement data models.",
        "Analyze data trends and patterns.",
        "Collaborate with cross-functional teams."
      ],
      preferredQualifications: [
        "Experience with machine learning frameworks.",
        "Proficiency in Python or R.",
        "Strong statistical analysis skills."
      ],
      applicationDeadline: "2024-11-25"
    },
    {
     jobId: 28,
      jobTitle: "Project Manager",
      company: "Business Solutions",
      location: "Seattle, WA (Remote)",
      jobType: "Full-Time",
      salary: "$80,000 - $100,000",
      jobDescription: "Seeking an experienced Project Manager to oversee various projects and ensure timely delivery.",
      jobResponsibilities: [
        "Develop project plans and timelines.",
        "Coordinate with stakeholders.",
        "Monitor project progress and adjust as necessary."
      ],
      preferredQualifications: [
        "Experience with project management software (e.g., MS Project).",
        "Strong leadership and communication skills.",
        "PMP certification is a plus."
      ],
      applicationDeadline: "2024-11-30"
    },
    {
     jobId: 29,
      jobTitle: "UX/UI Designer",
      company: "Creative Minds",
      location: "Austin, TX (Remote)",
      jobType: "Full-Time",
      salary: "$75,000 - $95,000",
      jobDescription: "We need a UX/UI Designer to create intuitive and visually appealing user interfaces.",
      jobResponsibilities: [
        "Design user interfaces for web and mobile applications.",
        "Conduct user research and usability testing.",
        "Collaborate with product and development teams."
      ],
      preferredQualifications: [
        "Proficiency in design tools (Sketch, Adobe XD).",
        "Strong portfolio showcasing UX/UI design work.",
        "Experience with wireframing and prototyping."
      ],
      applicationDeadline: "2024-12-05"
    },
    {
     jobId: 30,
      jobTitle: "Systems Analyst",
      company: "Tech Innovators",
      location: "Phoenix, AZ (Remote)",
      jobType: "Full-Time",
      salary: "$70,000 - $90,000",
      jobDescription: "We need a Systems Analyst to implement, maintain, and support IT systems.",
      jobResponsibilities: [
        "Analyze and design IT solutions.",
        "Document system requirements and specifications.",
        "Collaborate with developers and end-users."
      ],
      preferredQualifications: [
        "Experience with SQL and software development.",
        "Strong analytical and problem-solving skills.",
        "Knowledge of business analysis techniques."
      ],
      applicationDeadline: "2024-12-10"
    },
    {
     jobId: 31,
      jobTitle: "Digital Marketing Specialist",
      company: "Marketing Pros",
      location: "Los Angeles, CA (Remote)",
      jobType: "Full-Time",
      salary: "$65,000 - $85,000",
      jobDescription: "Looking for a Digital Marketing Specialist to develop and execute marketing campaigns.",
      jobResponsibilities: [
        "Create and manage digital marketing campaigns.",
        "Analyze and report on campaign performance.",
        "Collaborate with the marketing team."
      ],
      preferredQualifications: [
        "Experience with SEO and SEM.",
        "Proficiency in Google Analytics and AdWords.",
        "Strong understanding of social media marketing."
      ],
      applicationDeadline: "2024-12-15"
    },
    {
     jobId: 32,
      jobTitle: "Network Administrator",
      company: "IT Solutions",
      location: "Miami, FL (On-site)",
      jobType: "Full-Time",
      salary: "$60,000 - $80,000",
      jobDescription: "Seeking a Network Administrator to maintain and optimize our network infrastructure.",
      jobResponsibilities: [
        "Install and configure network hardware and software.",
        "Monitor network performance and troubleshoot issues.",
        "Ensure network security and compliance."
      ],
      preferredQualifications: [
        "Experience with networking protocols and services.",
        "Knowledge of firewalls and security systems.",
        "Certification (CCNA, CCNP) is a plus."
      ],
      applicationDeadline: "2024-12-20"
    },
    {
     jobId: 33,
      jobTitle: "Financial Analyst",
      company: "Finance Experts",
      location: "Houston, TX (Remote)",
      jobType: "Full-Time",
      salary: "$70,000 - $95,000",
      jobDescription: "Looking for a Financial Analyst to provide financial planning and analysis.",
      jobResponsibilities: [
        "Analyze financial data and create financial models.",
        "Prepare reports and forecasts.",
        "Assist in budgeting and financial planning."
      ],
      preferredQualifications: [
        "Experience with financial analysis and modeling.",
        "Proficiency in Excel and financial software.",
        "Strong analytical and problem-solving skills."
      ],
      applicationDeadline: "2024-12-25"
    },
    {
     jobId: 34,
      jobTitle: "Software Engineer",
      company: "Tech Solutions",
      location: "New York, NY (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "We are looking for a Software Engineer to develop and maintain software applications.",
      jobResponsibilities: [
        "Write clean and efficient code.",
        "Collaborate with cross-functional teams.",
        "Troubleshoot and debug applications."
      ],
      preferredQualifications: [
        "Experience with software development lifecycle.",
        "Proficiency in programming languages (e.g., Java, C++).",
        "Strong problem-solving skills."
      ],
      applicationDeadline: "2024-12-30"
    },
    {
     jobId: 35,
      jobTitle: "Human Resources Manager",
      company: "HR Pros",
      location: "Boston, MA (Remote)",
      jobType: "Full-Time",
      salary: "$80,000 - $100,000",
      jobDescription: "Seeking an HR Manager to oversee HR policies and procedures.",
      jobResponsibilities: [
        "Develop and implement HR strategies.",
        "Manage recruitment and selection processes.",
        "Oversee employee relations and performance management."
      ],
      preferredQualifications: [
        "Experience in HR management.",
        "Strong knowledge of labor laws and regulations.",
        "Excellent communication and interpersonal skills."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId:36,
      jobTitle: "Graphic Designer",
      company: "Creative Studio",
      location: "San Diego, CA (Remote)",
      jobType: "Full-Time",
      salary: "$65,000 - $85,000",
      jobDescription: "We need a Graphic Designer to create visually appealing graphics and layouts.",
      jobResponsibilities: [
        "Design graphics for digital and print media.",
        "Collaborate with marketing and design teams.",
        "Ensure brand consistency in all designs."
      ],
      preferredQualifications: [
        "Proficiency in Adobe Creative Suite.",
        "Strong portfolio showcasing design work.",
        "Experience with typography and color theory."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId:37,
      jobTitle: "Cybersecurity Analyst",
      company: "SecureTech",
      location: "Washington, D.C. (Remote)",
      jobType: "Full-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "Looking for a Cybersecurity Analyst to monitor and protect our systems.",
      jobResponsibilities: [
        "Monitor network for security breaches.",
        "Investigate security incidents.",
        "Implement security measures and protocols."
      ],
      preferredQualifications: [
        "Experience with SIEM tools.",
        "Knowledge of cybersecurity frameworks.",
        "Strong analytical and problem-solving skills."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 38,
      jobTitle: "Cloud Solutions Architect",
      company: "Cloud Experts",
      location: "Seattle, WA (Remote)",
      jobType: "Full-Time",
      salary: "$100,000 - $130,000",
      jobDescription: "Seeking a Cloud Solutions Architect to design and implement cloud solutions.",
      jobResponsibilities: [
        "Develop cloud strategies and roadmaps.",
        "Design cloud architectures.",
        "Ensure cloud solutions are secure and scalable."
      ],
      preferredQualifications: [
        "Experience with AWS, Azure, or GCP.",
        "Knowledge of cloud security and compliance.",
        "Strong understanding of cloud computing concepts."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 39,
      jobTitle: "Product Manager",
      company: "Innovative Products",
      location: "Boston, MA (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "Looking for a Product Manager to oversee the development and delivery of our products.",
      jobResponsibilities: [
        "Define product vision and strategy.",
        "Manage product lifecycle.",
        "Collaborate with cross-functional teams."
      ],
      preferredQualifications: [
        "Experience in product management.",
        "Strong understanding of Agile methodologies.",
        "Excellent communication and leadership skills."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 40,
      jobTitle: "Content Writer",
      company: "Content Creators",
      location: "Los Angeles, CA (Remote)",
      jobType: "Full-Time",
      salary: "$55,000 - $75,000",
      jobDescription: "We need a Content Writer to create engaging and informative content.",
      jobResponsibilities: [
        "Write articles, blog posts, and social media content.",
        "Research and develop content ideas.",
        "Collaborate with marketing and design teams."
      ],
      preferredQualifications: [
        "Strong writing and editing skills.",
        "Experience with SEO best practices.",
        "Proficiency in content management systems."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 41,
      jobTitle: "Sales Executive",
      company: "Sales Pros",
      location: "Houston, TX (Remote)",
      jobType: "Full-Time",
      salary: "$70,000 - $90,000",
      jobDescription: "Looking for a Sales Executive to drive sales and revenue growth.",
      jobResponsibilities: [
        "Identify and develop new business opportunities.",
        "Manage and grow client relationships.",
        "Achieve sales targets and goals."
      ],
      preferredQualifications: [
        "Experience in sales and business development.",
        "Strong negotiation and communication skills.",
        "Proficiency in CRM software."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 42,
      jobTitle: "Business Intelligence Analyst",
      company: "BizTech",
      location: "Houston, TX (Remote)",
      jobType: "Full-Time",
      salary: "$80,000 - $105,000",
      jobDescription: "We need a Business Intelligence Analyst to assess business processes and provide data-driven insights.",
      jobResponsibilities: [
        "Develop and maintain BI solutions.",
        "Analyze data and generate reports.",
        "Collaborate with business stakeholders."
      ],
      preferredQualifications: [
        "Experience with BI tools (e.g., Tableau, Power BI).",
        "Strong analytical and problem-solving skills.",
        "Proficiency in SQL and data analysis."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 43,
      jobTitle: "Customer Support Specialist",
      company: "HelpDesk Heroes",
      location: "Miami, FL (On-site)",
      jobType: "Full-Time",
      salary: "$50,000 - $65,000",
      jobDescription: "Seeking a Customer Support Specialist to provide technical assistance to computer users.",
      jobResponsibilities: [
        "Respond to customer inquiries and issues.",
        "Troubleshoot and resolve technical problems.",
        "Document support interactions and solutions."
      ],
      preferredQualifications: [
        "Experience in customer support or technical support.",
        "Strong communication and problem-solving skills.",
        "Knowledge of computer hardware and software."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 44,
      jobTitle: "Operations Manager",
      company: "Business Solutions",
      location: "Seattle, WA (Remote)",
      jobType: "Full-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "We need an Operations Manager to oversee daily operations and ensure efficiency.",
      jobResponsibilities: [
        "Manage operational processes and workflows.",
        "Develop and implement operational strategies.",
        "Monitor and improve performance metrics."
      ],
      preferredQualifications: [
        "Experience in operations management.",
        "Strong organizational and leadership skills.",
        "Knowledge of process improvement methodologies."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 45,
      jobTitle: "SEO Specialist",
      company: "Marketing Pros",
      location: "Los Angeles, CA (Remote)",
      jobType: "Full-Time",
      salary: "$60,000 - $80,000",
      jobDescription: "Looking for an SEO Specialist to improve our website's search engine ranking.",
      jobResponsibilities: [
        "Conduct keyword research and analysis.",
        "Optimize website content and structure.",
        "Monitor and report on SEO performance."
      ],
      preferredQualifications: [
        "Experience with SEO tools (e.g., SEMrush, Ahrefs).",
        "Strong understanding of search engine algorithms.",
        "Proficiency in HTML and web analytics."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 46,
      jobTitle: "Mobile App Developer",
      company: "App Masters",
      location: "New York, NY (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $115,000",
      jobDescription: "Seeking a Mobile App Developer to create high-quality mobile applications.",
      jobResponsibilities: [
        "Develop mobile apps for Android and iOS.",
        "Collaborate with designers and developers.",
        "Ensure app performance and stability."
      ],
      preferredQualifications: [
        "Proficiency in Swift and Kotlin.",
        "Experience with React Native.",
        "Strong understanding of mobile development frameworks."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 47,
      jobTitle: "MERN Stack Developer",
      company: "Tech Innovators",
      location: "San Francisco, CA (Remote)",
      jobType: "Full-Time",
      salary: "$90,000 - $120,000",
      jobDescription: "Looking for a MERN Stack Developer to build dynamic web applications.",
      jobResponsibilities: [
        "Develop front-end using React.js.",
        "Design and manage database with MongoDB.",
        "Implement server-side logic with Node.js and Express.js."
      ],
      preferredQualifications: [
        "Experience with RESTful APIs.",
        "Knowledge of cloud services (AWS/GCP).",
        "Strong understanding of CI/CD pipelines."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId:48,
      jobTitle: "Full Stack Developer 3",
      company: "Innovative Solutions",
      location: "Boston",
      jobType: "Full-Time",
      salary: "$100,000 - $130,000",
      jobDescription: "We need a Full Stack Developer experienced in both front-end and back-end development.",
      jobResponsibilities: [
        "Develop and maintain web applications.",
        "Collaborate with designers and other developers.",
        "Ensure responsive design and cross-browser compatibility."
      ],
      preferredQualifications: [
        "Experience with JavaScript frameworks.",
        "Knowledge of database management (SQL/NoSQL).",
        "Familiarity with version control systems (Git)."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 49,
      jobTitle: "Python Developer",
      company: "Data Wizards",
      location: "Boston, MA (Remote)",
      jobType: "Part-Time",
      salary: "$85,000 - $110,000",
      jobDescription: "Seeking a skilled Python Developer to develop and maintain our software solutions.",
      jobResponsibilities: [
        "Write reusable, testable, and efficient code.",
        "Design and implement low-latency, high-availability applications.",
        "Integrate user-facing elements with server-side logic."
      ],
      preferredQualifications: [
        "Familiarity with front-end technologies (like JavaScript and HTML5).",
        "Understanding of accessibility and security compliance.",
        "Knowledge of user authentication and authorization."
      ],
      applicationDeadline: "2024-12-31"
    },
    {
     jobId: 50,
      jobTitle: "Data Scientist-3",
      company: "Data Insights",
      location: "Chicago, IL (Remote)",
      jobType: "Full-Time",
      salary: "$95,000 - $125,000",
      jobDescription: "We are looking for a Data Scientist to analyze large datasets and derive actionable insights.",
      jobResponsibilities: [
        "Develop and implement data models.",
        "Analyze data trends and patterns.",
        "Collaborate with cross-functional teams."
      ],
      preferredQualifications: [
        "Experience with machine learning frameworks.",
        "Proficiency in Python or R.",
        "Strong statistical analysis skills."
      ],
      applicationDeadline: "2024-12-31"
    },
  ];

  module.exports=jobPostings;
  