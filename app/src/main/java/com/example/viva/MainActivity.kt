package com.example.viva

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.jlibrosa.audio.JLibrosa
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.IntBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.Random
import kotlin.math.ln
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var numQuestionsEditText: EditText
    private lateinit var generateButton: Button
    private lateinit var questionsTextView: TextView
    private lateinit var subjectSpinner: Spinner
    lateinit var textToSpeech: TextToSpeech
    private lateinit var resultTextView: TextView
    private lateinit var scoreTextView: TextView
    private lateinit var Button: Button
    private lateinit var scoreButton: Button
    private lateinit var mediaRecorder: MediaRecorder
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var modelInterpreter: Interpreter? = null



    private val subjectFiles = mapOf(
        "Operating System" to R.raw.questions_os,
        "Computer Network" to R.raw.questions_cn
        // Add more subjects and corresponding file paths as needed
    )

    private var currentQuestionIndex = 0
    private val selectedQuestions = mutableListOf<String>()
    private val ttsTimeout = 180000L // 3 minutes in milliseconds
    private lateinit var questionTimer: CountDownTimer
    private lateinit var tfLiteModel: MappedByteBuffer
    private lateinit var tfLite: Interpreter
    private lateinit var transcribeButton: Button
    private lateinit var playAudioButton: Button
    private lateinit var resultTextview: TextView

    private val mediaPlayer = MediaPlayer()

    private val TAG = "TfLiteASRDemo"
    private val SAMPLE_RATE = 16000
    private val DEFAULT_AUDIO_DURATION = -1
    private val wavFilename = "deep.wav"
    private val TFLITE_FILE = "model.tflite"


    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        numQuestionsEditText = findViewById(R.id.numQuestionsEditText)
        playAudioButton = findViewById(R.id.playAudioButton)
        generateButton = findViewById(R.id.generateButton)
        questionsTextView = findViewById(R.id.questionsTextView)
        subjectSpinner = findViewById(R.id.subjectSpinner)
        resultTextView = findViewById(R.id.resultTextView)
        scoreTextView=findViewById(R.id.scoreTextView)
        scoreButton=findViewById(R.id.viewScoreButton)


//        val doc2 = "reading something about life no one else knows"
//        val doc3 = "Never stop learning"


        scoreButton.setOnClickListener{
            val doc1 = resultTextView.text.toString()
            val query = "Operating System is a software"
            val tfDoc1 = computeTF(doc1)
            val idfDict = computeIDF(listOf(doc1))
            val tfDoc1Normalized = computeNormalizedTF(doc1)
            val tfidfDoc1 = computeTFIDF(tfDoc1Normalized, idfDict)

            val tfQueryNormalized = computeNormalizedTF(query)
            val idfQuery = computeIDF(listOf(query))
            val tfidfQuery = computeTFIDF(tfQueryNormalized, idfQuery)
            Log.d("TF-IDFscore","$tfidfDoc1")
            Log.d("TF-IDFscorequery","$tfidfQuery")
            val similarity = cosineSimilarity(tfidfQuery, tfidfDoc1)
            Log.d("Cosine","$similarity")
            scoreTextView.text=similarity.toString()

        }



        val jLibrosa = JLibrosa()



        // Initialize the TextToSpeech object
        textToSpeech = TextToSpeech(this, this)




        playAudioButton.setOnClickListener {
            try {
                assets.openFd(wavFilename).use { assetFileDescriptor ->
                    mediaPlayer.reset()
                    mediaPlayer.setDataSource(
                        assetFileDescriptor.fileDescriptor,
                        assetFileDescriptor.startOffset,
                        assetFileDescriptor.length
                    )
                    mediaPlayer.prepare()
                }
            } catch (e: Exception) {
                Log.e(TAG, e.message ?: "Error playing audio")
            }
            mediaPlayer.start()
        }

        transcribeButton = findViewById(R.id.recognizeButton)
        resultTextview = findViewById(R.id.resultTextView)
        transcribeButton.setOnClickListener {
            try {
                val audioFeatureValues = jLibrosa.loadAndRead(
                    copyWavFileToCache(wavFilename),
                    SAMPLE_RATE,
                    DEFAULT_AUDIO_DURATION
                )

                val inputArray = arrayOf(audioFeatureValues)
                val outputBuffer = IntBuffer.allocate(2000)

                val outputMap: MutableMap<Int, Any> = HashMap()
                outputMap[0] = outputBuffer

                tfLiteModel = loadModelFile(assets, TFLITE_FILE)
                val tfLiteOptions = Interpreter.Options()
                tfLite = Interpreter(tfLiteModel, tfLiteOptions)
                tfLite.resizeInput(0, intArrayOf(audioFeatureValues.size))

                tfLite.runForMultipleInputsOutputs(inputArray, outputMap)

                val outputSize = tfLite.getOutputTensor(0).shape()[0]
                val outputArray = IntArray(outputSize)
                outputBuffer.rewind()
                outputBuffer.get(outputArray)
                val finalResult = StringBuilder()
                for (i in 0 until outputSize) {
                    val c = outputArray[i].toChar()
                    if (outputArray[i] != 0) {
                        finalResult.append(outputArray[i].toChar())
                    }
                }
                resultTextview.text = finalResult.toString()
            } catch (e: Exception) {
                Log.e(TAG, e.message ?: "Error transcribing")
            }
        }
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                1
            )
        }
        mediaRecorder = MediaRecorder()

        // Populate the spinner with subject options
        val subjectOptions = resources.getStringArray(R.array.subject_options)
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, subjectOptions)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        subjectSpinner.adapter = adapter

        generateButton.setOnClickListener {
            generateRandomQuestionsInOrder()
        }

    }

@RequiresApi(Build.VERSION_CODES.N)
fun computeTF(document: String): Map<String, Int> {
    val wordMap = mutableMapOf<String, Int>()
    val docTokens = document.split(" ")
    for (token in docTokens) {
        wordMap[token] = wordMap.getOrDefault(token, 0) + 1
    }
    return wordMap
}

    fun termFrequency(term: String, document: String): Double {
        val normalizeDocument = document.toLowerCase().split(" ")
        return normalizeDocument.count { it == term.toLowerCase() } / normalizeDocument.size.toDouble()
    }
    fun inverseDocumentFrequency(term: String, allDocuments: List<String>): Double {
        val numDocumentsWithThisTerm = allDocuments.count { term.toLowerCase() in it.toLowerCase().split(" ") }
        return if (numDocumentsWithThisTerm > 0) {
            1.0 + ln(allDocuments.size.toDouble() / numDocumentsWithThisTerm)
        } else {
            1.0
        }
    }

    fun computeIDF(documents: List<String>): Map<String, Double> {
        val idfMap = mutableMapOf<String, Double>()
        for (doc in documents) {
            val sentence = doc.split(" ")
            for (word in sentence) {
                idfMap[word] = inverseDocumentFrequency(word, documents)
            }
        }
        return idfMap
    }

    fun computeNormalizedTF(document: String): Map<String, Double> {
        val normTF = mutableMapOf<String, Double>()
        val sentence = document.split(" ")
        for (word in sentence) {
            normTF[word] = termFrequency(word, document)
        }
        return normTF
    }
    fun computeTFIDF(tfDoc: Map<String, Double>, idfDict: Map<String, Double>): Map<String, Double> {
        val tfidfDict = mutableMapOf<String, Double>()
        for ((word, tf) in tfDoc) {
            tfidfDict[word] = tf * idfDict[word]!!
        }
        return tfidfDict
    }

    @RequiresApi(Build.VERSION_CODES.N)
    fun cosineSimilarity(tfidfQuery: Map<String, Double>, tfidfDoc: Map<String, Double>): Double {
        val dotProduct = (tfidfQuery.keys + tfidfDoc.keys).sumByDouble { tfidfQuery.getOrDefault(it, 0.0) * tfidfDoc.getOrDefault(it, 0.0) }
        val qryMod = sqrt(tfidfQuery.values.sumByDouble { it * it })
        val docMod = sqrt(tfidfDoc.values.sumByDouble { it * it })
        val denominator = qryMod * docMod
        return if (denominator == 0.0) 0.0 else dotProduct / denominator
    }


    private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun copyWavFileToCache(wavFilename: String): String {
        val destinationFile = File(cacheDir, wavFilename)
        if (!destinationFile.exists()) {
            try {
                assets.open(wavFilename).use { inputStream ->
                    val inputStreamSize = inputStream.available()
                    val buffer = ByteArray(inputStreamSize)
                    inputStream.read(buffer)
                    FileOutputStream(destinationFile).use { fileOutputStream ->
                        fileOutputStream.write(buffer)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, e.message ?: "Error copying WAV file to cache")
            }
        }
        return destinationFile.path
    }


    private fun startRecording(callback: (ByteArray) -> Unit) {
        try {
            mediaRecorder.apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
                setOutputFile("/dev/null")
                prepare()
                start()
            }
            isRecording = true
            playAudioButton.text = "Stop Recording"
            Toast.makeText(applicationContext, "Recording started", Toast.LENGTH_SHORT).show()

            callback(byteArrayOf())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun stopRecording() {
        mediaRecorder.apply {
            stop()
            release()
        }
        isRecording = false
        playAudioButton.text = "Start Recording"
        Toast.makeText(applicationContext, "Recording stopped", Toast.LENGTH_SHORT).show()
    }

    private fun processAudioData(audioData: ByteArray) {

    }


//    @RequiresApi(Build.VERSION_CODES.N)
//    private fun computeTF(docs: List<String>): List<MutableMap<String, Double>> {
//        val tfDocs = mutableListOf<MutableMap<String, Double>>()
//
//        for (doc in docs) {
//            val docWords = doc.split(" ")
//            val wordCount = docWords.size
//
//            val wordFreq = mutableMapOf<String, Int>()
//            for (word in docWords) {
//                wordFreq.put(word, wordFreq.getOrDefault(word, 0) + 1)
//            }
//
//            val tfData = mutableMapOf<String, Double>()
//            for ((word, freq) in wordFreq) {
//                tfData.put(word, freq.toDouble() / wordCount)
//            }
//            tfDocs.add(tfData)
//        }
//
//        return tfDocs
//    }
//    fun termFrequency(term: String, document: String): Double {
//        val normalizeDocument = document.toLowerCase(Locale.getDefault()).split(" ")
//        return normalizeDocument.filter { it == term.toLowerCase(Locale.getDefault()) }.count() / normalizeDocument.size.toDouble()
//    }
//
//    fun computeNormalizedTF(documents: List<String>): MutableList<MutableMap<String, Double>> {
//        val tfDoc = mutableListOf<MutableMap<String, Double>>()
//
//        for (doc in documents) {
//            val sentence = doc.split(" ")
//            val normTf = mutableMapOf<String, Double>()
//
//            for (word in sentence) {
//                normTf[word] = termFrequency(word, doc)
//            }
//
//            tfDoc.add(normTf)
//            // Displaying the Normalized TF DataFrame is omitted in this example
//        }
//
//        return tfDoc
//    }
//    fun inverseDocumentFrequency(term: String, allDocuments: List<String>): Double {
//        var numDocumentsWithThisTerm = 0
//
//        for (doc in allDocuments) {
//            if (doc.toLowerCase(Locale.getDefault()).contains(term.toLowerCase(Locale.getDefault()))) {
//                numDocumentsWithThisTerm++
//            }
//        }
//
//        return if (numDocumentsWithThisTerm > 0) {
//            1.0 + Math.log(allDocuments.size.toDouble() / numDocumentsWithThisTerm)
//        } else {
//            1.0
//        }
//    }
//
//    fun computeIDF(documents: List<String>): Map<String, Double> {
//        val idfDict = mutableMapOf<String, Double>()
//
//        for (doc in documents) {
//            val sentence = doc.split(" ")
//
//            for (word in sentence) {
//                idfDict[word] = inverseDocumentFrequency(word, documents)
//            }
//        }
//        for ((word, idf) in idfDict) {
//            Log.d("IDF Results", "IDF for term '$word': $idf")
//        }
//
//        return idfDict
//    }
//        fun computeTFIDFWithAllDocs(documents: List<String>, query: String): Pair<DoubleArray, MutableList<MutableList<Double>>> {
//            val idfDict = computeIDF(documents)
//            val tfDoc = computeNormalizedTF(documents)
//
//    //        val queryTokens = query.toLower   Case(Locale.getDefault()).split(" ")
//            val queryTokens = query.split(" ")
//
//            val df = mutableListOf<MutableList<Double>>()
//            val columns = mutableListOf<String>()
//            // Initialize the DataFrame with the given documents and query tokens
//            for (i in 0 until documents.size) {
//                val row = MutableList(queryTokens.size + 1) { 0.0 }
//                row[0] = i.toDouble()
//                df.add(row)
//
//            }
//            columns.add("doc")  // Add "doc" as the first column name
//            columns.addAll(queryTokens)
//
//            for ((index, doc) in documents.withIndex()) {
//                val sentence = doc.split(" ")
//
//                for (word in sentence) {
//                    if (word in queryTokens) {
//                        val wordIdx = queryTokens.indexOf(word)
//                        val tfIdfScore = tfDoc[index][word.toLowerCase(Locale.getDefault())]?.let { it * idfDict[word]!! } ?: 0.0
//                        df[index][wordIdx + 1] = tfIdfScore
//                    }
//                }
//            }
//            for ((idx, row) in df.withIndex()) {
//                Log.d("TF-IDF Results", "Document ${columns[idx]}: $row")
//            }
//
//            return Pair(df.flatten().toDoubleArray(), df)
//        }
//    fun computeQueryTF(query: String): Map<String, Double> {
//        val queryTokens = query.split(" ")
//        val queryNormTF = mutableMapOf<String, Double>()
//
//        for (word in queryTokens) {
//            queryNormTF[word] = termFrequency(word, query)
//        }
//
//        // Log the query normalized TF results
//        Log.d("Query TF Results", "Query Normalized TF: $queryNormTF")
//
//        return queryNormTF
//    }
//    fun computeQueryIDF(query: String, documents: List<String>): Map<String, Double> {
//        val queryTokens = query.split(" ")
//        val idfDictQry = mutableMapOf<String, Double>()
//
//        for (word in queryTokens) {
//            idfDictQry[word] = inverseDocumentFrequency(word, documents)
//        }
//
//        // Log the query IDF results
//        Log.d("Query IDF Results", "Query IDF: $idfDictQry")
//
//        return idfDictQry
//    }
//    fun computeQueryTFIDF(query: String, queryNormTF: Map<String, Double>, idfDictQry: Map<String, Double>): Map<String, Double> {
//        val tfIdfDictQry = mutableMapOf<String, Double>()
//
//        for (word in query.split(" ")) {
//            tfIdfDictQry[word] = queryNormTF[word]?.let { it * idfDictQry[word]!! } ?: 0.0
//        }
//
//        // Log the query TF-IDF results
//        Log.d("Query TF-IDF Results", "Query TF-IDF: $tfIdfDictQry")
//
//        return tfIdfDictQry
//    }
//    fun cosineSimilarity(tfidfDictQry: Map<String, Double>, df: Map<String, Map<String, Double>>, query: String, docNum: Int): Double {
//        var dotProduct = 0.0
//        var qryMod = 0.0
//        var docMod = 0.0
//        val tokens = query.split(" ")
//
//        for (keyword in tokens) {
//            Log.d("toggle","$keyword")
//            if ((df[docNum] as List<String>).contains(keyword)) {
//                val value = tfidfDictQry[keyword]!!
//                val keywordIndex = (df[docNum] as List<String>).indexOf(keyword)
//                dotProduct += value * df[docNum][keywordIndex]
//                qryMod += value * value
//                Log.d("Hello", "$qryMod")
//                docMod += df[docNum][keywordIndex] * df[docNum][keywordIndex]
//            }
//        }

//        qryMod = Math.sqrt(qryMod)
//        Log.d("qryMod","$qryMod")
//        docMod = Math.sqrt(docMod)
//        Log.d("docMod","$docMod")
//
//        val denominator = qryMod * docMod
//        Log.d("denominator","$denominator")
//        val cosSim = dotProduct / denominator
//
//        return cosSim
//    }

//    fun flatten(list: List<Any>): List<Any> {
//        val resultList = mutableListOf<Any>()
//        for (item in list) {
//            if (item is List<*> && item.isNotEmpty() && item[0] !is String) {
//                resultList.addAll(flatten(item as List<Any>))
//            } else {
//                resultList.add(item)
//            }
//        }
//        return resultList
//    }
//
//    fun rankSimilarityDocs(documents: List<String>): List<Double> {
//        val cosSim = mutableListOf<Double>()
//        val query = "life learning"
//        val (tfIdf, df) = computeTFIDFWithAllDocs(documents,query)
//        val queryTokens = query.toLowerCase(Locale.getDefault()).split(" ")
//        val queryNormTF = computeQueryTF(query)
//        val idfDictQry = computeQueryIDF(query,documents)
//        val queryTFIDF = computeQueryTFIDF(query, queryNormTF, idfDictQry)
////
//        for (docNum in documents.indices) {
//            cosSim.add(cosineSimilarity(queryTFIDF, df, query, docNum))
//        }
//        return cosSim
//    }

//    fun termFrequency(term: String, document: String): Double {
//        val normalizeDocument = document.lowercase().split(" ")
//        return normalizeDocument.count { it == term.lowercase() } / normalizeDocument.size.toDouble()
//    }
//
//    fun inverseDocumentFrequency(term: String, allDocuments: List<String>): Double {
//        val numDocumentsWithThisTerm = allDocuments.count { term.lowercase() in it.lowercase().split(" ") }
//        return if (numDocumentsWithThisTerm > 0) {
//            1.0 + log(allDocuments.size.toDouble() / numDocumentsWithThisTerm)
//        } else {
//            1.0
//        }
//    }
//
//    fun computeTFIDF(doc: String, query: String): Double {
//        val tf = termFrequency(query, doc)
//        val idf = inverseDocumentFrequency(query, listOf(doc))
//        return tf * idf
//    }
//
//    fun cosineSimilarity(tfidfQuery: Double, tfidfDoc: Double): Double {
//        return if (tfidfQuery != 0.0 && tfidfDoc != 0.0) {
//            (tfidfQuery * tfidfDoc) / (sqrt(tfidfQuery * tfidfQuery) * sqrt(tfidfDoc * tfidfDoc))
//        } else {
//            0.0
//        }
//    }






    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {

            val language = Locale.US
            val result = textToSpeech.setLanguage(language)

            if (result == TextToSpeech.LANG_MISSING_DATA ||
                result == TextToSpeech.LANG_NOT_SUPPORTED
            ) {
            } else {
                val slowRate = 0.7f
                val normalPitch = 1.0f
                textToSpeech.setSpeechRate(slowRate)
                textToSpeech.setPitch(normalPitch)
            }
        } else {
        }
    }

    private fun askNextQuestion() {
        if (currentQuestionIndex < selectedQuestions.size) {
            val questionToAsk = selectedQuestions[currentQuestionIndex]
            textToSpeech.speak(questionToAsk, TextToSpeech.QUEUE_FLUSH, null, null)
            currentQuestionIndex++

            startQuestionTimer()
        } else {
            currentQuestionIndex = 0
        }
    }

    private fun startQuestionTimer() {
        questionTimer = object : CountDownTimer(60000, 1000) {
            override fun onTick(millisUntilFinished: Long) {
            }

            override fun onFinish() {
                askNextQuestion()
            }
        }
        questionTimer.start()
    }
//    private fun tokenize(text: String): List<String> {
//        // Tokenize the text into lowercase words using split
//        return text.split("\\s+")
//    }
//    private fun tfIdfVectorize(tokens: List<String>, vocabulary: List<String>): RealVector {
//        val vector = ArrayRealVector(vocabulary.size)
//        val documentCount = 2 // Assuming we have 2 documents (string1 and string2)
//
//        // Calculate term frequency (TF) for each word in the document
//        val termFrequencies = tokens.groupBy { it }.mapValues { group -> group.value.count() }
//
//        for (word in vocabulary) {
//            val tf = termFrequencies[word]?.toDouble() ?: 0.0
//            val idf = Math.log(documentCount.toDouble() / (vocabulary.count { it == word } + 1))
//            vector.setEntry(vocabulary.indexOf(word), tf * idf)
//        }
//
//        return vector
//    }
//    private fun calculateCosineSimilarity(vector1: RealVector, vector2: RealVector): Double {
//        val dotProduct = vector1.dotProduct(vector2)
////        val norm1 = vector1.getNorm()
////        val norm2 = vector2.getNorm()
////
////        if (norm1 > 0 && norm2 > 0) {
////            val dotProduct = vector1.dotProduct(vector2)
////            return dotProduct / (norm1 * norm2)
////        } else {
////            // Handle the case where at least one vector has zero norm
////            return 0.0
////        }
//
//        val euclideanDistance = Math.sqrt(EuclideanDistance.compute(vector1, vector2))
//
//        if (euclideanDistance > 0) {
//            return dotProduct / (euclideanDistance * euclideanDistance) // Normalize by squared distance
//        } else {
//            // Handle the case where at least one vector has zero norm
//            return 0.0
//        }
//
//    }
//    private fun calculateTFIDFCosineSimilarity(string1: String, string2: String): Double {
//        // Tokenize the strings into words (lowercase and remove punctuation)
//        val tokens1 = tokenize(string1.toLowerCase().replace("[^\\w\\s]".toRegex(), ""))
//        val tokens2 = tokenize(string2.toLowerCase().replace("[^\\w\\s]".toRegex(), ""))
//
//        // Create a combined vocabulary of unique words
//        val vocabulary = (tokens1 + tokens2).distinct()
//
//        // Create TF-IDF vectors for each string
//        val vector1 = tfIdfVectorize(tokens1, vocabulary)
//        val vector2 = tfIdfVectorize(tokens2, vocabulary)
//
//        // Calculate cosine similarity using Apache Commons Math library
//        val similarity = calculateCosineSimilarity(vector1, vector2)
//
//        return similarity
//    }
    private fun generateRandomQuestionsInOrder() {
        val selectedSubject = subjectSpinner.selectedItem.toString()
        val numQuestions = numQuestionsEditText.text.toString().toIntOrNull()

        val fileId = subjectFiles[selectedSubject]

        if (fileId != null && numQuestions != null && numQuestions > 0) {
            val random = Random()

            val inputStream = resources.openRawResource(fileId)
            val reader = BufferedReader(InputStreamReader(inputStream))
            val allQuestions = reader.readLines()

            while (selectedQuestions.size < numQuestions) {
                val randomIndex = random.nextInt(allQuestions.size)
                val selectedQuestion = allQuestions[randomIndex]
                selectedQuestions.add("${selectedQuestions.size + 1}. $selectedQuestion")
            }

            val selectedQuestionsString = selectedQuestions.joinToString("\n")
            questionsTextView.text = selectedQuestionsString

            askNextQuestion()
        } else {
            questionsTextView.text = "Invalid input or not enough questions available."
        }
    }

    override fun onDestroy() {
        if (textToSpeech.isSpeaking) {
            textToSpeech.stop()
        }
        textToSpeech.shutdown()

        if (::questionTimer.isInitialized) {
            questionTimer.cancel()
        }

        super.onDestroy()
    }}

