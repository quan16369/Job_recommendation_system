const express = require("express");
const multer = require("multer");
const { ChromaClient } = require("chromadb");
const { HfInference } = require("@huggingface/inference");
const fs = require("fs");
const path = require("path");
const pdf = require("pdf-parse");
const jobPostings = require("./jobPostings.js"); // Data for job postings
const client = new ChromaClient();
const hf = new HfInference(process.env.HF_API_KEY);
const collectionName = "job_collection";
require("dotenv").config();

const app = express();
const port = 3000;

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "./uploads"); // Specify directory to save files
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname)); // Unique filename based on current timestamp
  },
});
const upload = multer({ storage: storage });

// Middleware to use EJS and serve static files from the 'public' folder
app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(express.urlencoded({ extended: true })); // Parse form data

// Function to generate embeddings for input texts
async function generateEmbeddings(texts) {
  const results = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2", // Model to generate sentence embeddings
    inputs: texts,
  });
  return results;
}

// Function to classify text into labels such as location, job title, etc.
async function classifyText(text, labels) {
  const response = await hf.request({
    model: "facebook/bart-large-mnli", // Model for multi-label classification
    inputs: text,
    parameters: { candidate_labels: labels },
  });
  return response;
}

// Function to extract filter criteria (location, job title, etc.) from query
async function extractFilterCriteria(query) {
  const criteria = {
    location: null,
    jobTitle: null,
    jobType: null,
    company: null,
    salary: null,
  };
  const labels = ["location", "job title", "company", "job type", "salary"];

  const words = query.split(" "); // Split query into words
  for (const word of words) {
    const result = await classifyText(word, labels); // Classify each word into the defined labels
    const highestScoreLabel = result.labels[0];
    const score = result.scores[0];
    if (score > 0.5) {
      // If score is above a threshold, assign it to a label
      switch (highestScoreLabel) {
        case "location":
          criteria.location = word;
          break;
        case "job title":
          criteria.jobTitle = word;
          break;
        case "company":
          criteria.company = word;
          break;
        case "job type":
          criteria.jobType = word;
          break;
        case "salary":
          criteria.salary = word;
          break;
        default:
          break;
      }
    }
  }
  return criteria;
}

// Function to perform similarity search using embeddings and ChromaDB
async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
  const queryEmbedding = await generateEmbeddings([queryTerm]); // Generate embedding for the query
  const results = await collection.query({
    collection: collectionName,
    queryEmbeddings: queryEmbedding,
    n: 3, // Limit results to top 3
  });

  if (!results || results.length === 0) {
    return [];
  }

  let topJobPostings = results.ids[0]
    .map((id, index) => {
      const job = jobPostings.find((job) => job.jobId.toString() === id);
      if (!job) return null;
      return {
        id,
        score: results.distances[0][index], // Distance score from the query
        job_title: job.jobTitle,
        job_description: job.jobDescription,
        job_type: job.jobType,
        company: job.company,
        location: job.location,
        salary: job.salary,
        job_responsibilities: job.jobResponsibilities,
        preferred_qualifications: job.preferredQualifications,
        application_deadline: job.applicationDeadline,
      };
    })
    .filter(Boolean); // Remove null values if any job is not found

  return topJobPostings.sort((a, b) => a.score - b.score); // Sort results by score
}

// Function to extract text from PDF file
const extractTextFromPDF = async (filePath) => {
  try {
    const dataBuffer = fs.readFileSync(filePath); // Read the file
    const data = await pdf(dataBuffer); // Extract text from the PDF
    const text = data.text.replace(/\n/g, " ").replace(/ +/g, " "); // Clean the extracted text
    return text;
  } catch (err) {
    console.error("Error extracting text from PDF:", err);
    throw err;
  }
};

// Route to render the homepage with search form
app.get("/", (req, res) => {
  res.render("index", { jobs: [], query: "" });
});

// Route to handle the search form submission and return search results
app.post("/search", async (req, res) => {
  const query = req.body.query;
  const collection = await client.getOrCreateCollection({
    name: collectionName,
  });

  // Handle potential jobId duplicates
  const uniqueIds = new Set();
  jobPostings.forEach((job, index) => {
    while (uniqueIds.has(job.jobId.toString())) {
      job.jobId = `${job.jobId}_${index}`; // Create unique jobId
    }
    uniqueIds.add(job.jobId.toString());
  });

  const jobTexts = jobPostings.map(
    (job) => `${job.jobTitle}. ${job.jobDescription}. ${job.jobType}`
  );

  // Save embeddings into ChromaDB collection
  const embeddingData = await generateEmbeddings(jobTexts);
  await collection.add({
    ids: jobPostings.map((job) => job.jobId.toString()),
    documents: jobTexts,
    embeddings: embeddingData,
  });

  // Extract filter criteria from the query
  const filterCriteria = await extractFilterCriteria(query);

  // Perform similarity search for relevant jobs
  const results = await performSimilaritySearch(
    collection,
    query,
    filterCriteria
  );

  // Render the search results
  res.render("index", { jobs: results, query: query });
});

// Route to render the upload page
app.get("/upload.html", (req, res) => {
  res.render("upload"); // Render the upload.ejs page
});

// Route to handle the uploaded PDF and return matching job postings
app.post("/upload-pdf", upload.single("resume"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }

  const filePath = path.join(__dirname, "uploads", req.file.filename); // Path to uploaded file

  try {
    const resumeText = await extractTextFromPDF(filePath); // Extract text from the uploaded resume

    // Generate embeddings for the resume text
    const resumeEmbedding = await generateEmbeddings([resumeText]);

    // Perform similarity search in the ChromaDB collection
    const collection = await client.getOrCreateCollection({
      name: collectionName,
    });
    const results = await collection.query({
      queryEmbeddings: resumeEmbedding,
      n: 3, // Top 3 matching job postings
    });

    if (results.ids.length > 0) {
      let topJobPostings = results.ids[0].map((id, index) => {
        const job = jobPostings.find((job) => job.jobId.toString() === id);
        return {
          id,
          score: results.distances[0][index],
          company: job.company,
          job_title: job.jobTitle,
          location: job.location,
          job_type: job.jobType,
          salary: job.salary,
          job_description: job.jobDescription,
          jobResponsibilities: job.jobResponsibilities,
          preferred_qualifications: job.preferredQualifications,
          application_deadline: job.applicationDeadline,
        };
      });
      res.render("index", { jobs: topJobPostings, query: "" });
    } else {
      res.render("index", { jobs: [], query: "No matching jobs found." });
    }
  } catch (err) {
    console.error("Error processing PDF:", err);
    res.render("index", { jobs: [], query: "Error processing PDF" });
  }
});

// Start the server on port 3000
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
