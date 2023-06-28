import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Sudi is a form of intelligence model that tracks what word represents what part of speech in different ways: console, file-testing
 * and hard-coding.
 *
 * @author Aryan Dawer, Dartmouth CS 10, Fall 2022
 * @author Lindsey Drumm, Dartmouth CS 10, Fall 2022
 *
 */

public class partOfSpeechTagger {

    private static Map<String, Map<String, Double>> tMap; // Transition map: Maps tag to all tags it maps to in training dataset
                                                          // and gives their probability {tag1 -> {tag2 -> prob, tag3 -> prob,...},...}

    private static Map<String, Map<String, Double>> eMap; // Emission map: Maps tag to all words it maps to in training dataset
                                                          // and gives their probability {tag1 -> {word1 -> prob, word2 -> prob,...},...}

    /**
     * Method that reads a file with new sentences on each line and makes a list of all words in the all sentences with
     * "#" as the separator for each sentence.
     * @param pathName name of the file to be iterated for words/tags
     * @return List of all words/tags seperated by "#" for new sentence
     */
    public static List<List<String>> getTrainingLists(String pathName) throws Exception{
        List<List<String>> trainingWords = new ArrayList<>();
        BufferedReader inp = null;
        String pathStart = "/Users/aryandawer/Documents/IdeaProjects/cs10/cs10/ps5/texts/";
        try{
            inp = new BufferedReader(new FileReader(pathStart+pathName));
            String sentence = null;
            while((sentence = inp.readLine())!=null){
                trainingWords.add(new ArrayList<>());
                trainingWords.get(trainingWords.size()-1).add("#");
                String[] words = sentence.split(" ");
                for(int i=0;i<words.length;i++){
                    trainingWords.get(trainingWords.size()-1).add(words[i].toLowerCase());
                }
            }
        }
        catch(Exception e){
            System.out.println("File could not be opened!");
        }
        finally{
            try{
                inp.close();
            }
            catch(IOException e){
                System.out.println("File could not be closed!");
            }
        }
        return trainingWords;
    }

    /**
     * Step within training the model: Takes either the transition map or emission map and normalizes the values and
     * calculates their log for easier calculation.
     * @param originalMap the map whose values need to be normalised and then taken
     *                    natural log of to make calculations additions instead of multiplications
     * @return the resultant map
     */
    public static Map<String, Map<String, Double>> normalizeAndLog(Map<String, Map<String, Double>> originalMap){
        for(Map<String, Double> tagFreqMap:originalMap.values()){
            double total = tagFreqMap.values().stream().mapToDouble(Double::doubleValue).sum();
            for(String tag:tagFreqMap.keySet()){
                tagFreqMap.put(tag, Math.log(tagFreqMap.get(tag)/total));
            }
        }
        return originalMap;
    }

    /**
     * Writes a method to train a model (observation and transition probabilities) on
     * corresponding lines (sentence and tags) from a pair of training files converted into
     * ArrayList of ArrayLists of strings forming sentences
     * @param words list of lists of words in a sentence
     * @param tags list of lists of tags denoting to parts-of-speech in the corresponding index sentence
     */
    public static void getTransitionAndEmissionMaps(List<List<String>> words, List<List<String>> tags){
        tMap = new HashMap<>();
        eMap = new HashMap<>();

        for (int i = 0; i < tags.size(); i++) {
            for (int j = 0; j < tags.get(i).size(); j++) {

                //for tMap
                if(!(j==tags.get(i).size()-1)){
                    if(!tMap.containsKey(tags.get(i).get(j))){
                        tMap.put(tags.get(i).get(j),new HashMap<>());
                    }
                    Map<String, Double> valUpdate_tMap = tMap.get(tags.get(i).get(j));
                    if(!valUpdate_tMap.containsKey(tags.get(i).get(j+1))){
                        valUpdate_tMap.put(tags.get(i).get(j+1),0.0);
                    }
                    valUpdate_tMap.put(tags.get(i).get(j+1),valUpdate_tMap.get(tags.get(i).get(j+1))+1.0);
                }

                // For eMap:
                if(!eMap.containsKey(tags.get(i).get(j))){
                    eMap.put(tags.get(i).get(j), new HashMap<>());
                }
                Map<String, Double> valUpdate_eMap = eMap.get(tags.get(i).get(j));
                if(!valUpdate_eMap.containsKey(words.get(i).get(j))){
                    valUpdate_eMap.put(words.get(i).get(j), 0.0);
                }
                valUpdate_eMap.put(words.get(i).get(j),valUpdate_eMap.get(words.get(i).get(j))+1.0);
            }
        }
        tMap = normalizeAndLog(tMap);
        eMap = normalizeAndLog(eMap);

    }

    /**
     * A method to perform Viterbi decoding to find the best sequence of tags for a line (sequence of words).
     * @param sentence Arraylist of given sentence to perform viterbi on
     * @return
     */
    public static List<String> doViterbi(List<String> sentence){
        List<Map<String,String>> backTrace = new ArrayList<>();
        Set<String> currTags = new HashSet<>();
        Map<String, Double> currScores = new HashMap<>();

        double unseen = -100.00;

        currTags.add("#");
        currScores.put("#", 0.0);

        for (int i = 0; i < sentence.size()-1; i++) {
            Set<String> nextTags = new HashSet<>();
            Map<String, Double> nextScores = new HashMap<>();
            Map<String,String> backTraceCurrStep = new HashMap<>();
            for(String currTag:currTags){
                if(tMap.get(currTag)!=null){
                    for(String nextTag : tMap.get(currTag).keySet()){
                        nextTags.add(nextTag);
                        double tScore = tMap.get(currTag).get(nextTag);
                        double oScore;
                        if(eMap.get(nextTag).containsKey(sentence.get(i+1))){
                            oScore = eMap.get(nextTag).get(sentence.get(i+1));
                        }
                        else{
                            oScore = unseen;
                        }
                        double nextScore = currScores.get(currTag) + tScore + oScore;
                        if(!(nextScores.containsKey(nextTag)) || nextScore>nextScores.get(nextTag)){
                            nextScores.put(nextTag, nextScore);
                            backTraceCurrStep.put(nextTag, currTag);
                        }
                    }
                }
            }
            backTrace.add(backTraceCurrStep);
            currTags = nextTags;
            currScores = nextScores;
        }

        String biggestScore = null;

        for (String k: currScores.keySet()) {
            if (biggestScore == null) {
                biggestScore = k;
            } else if (currScores.get(k) > currScores.get(biggestScore)) {
                biggestScore = k;
            }
        }

        List<String> path = new ArrayList<>();
        String current = biggestScore;
        for (int i = backTrace.size() - 1; i > -1; i--) {
            path.add(0, current);
            current = backTrace.get(i).get(current);
        }
        path.add(0, current);
        return path;
    }

    /**
     * Runs viterbi on each testing sentence and then checks similarity of output with all tags in given testing tags.
     * Keeps count of number of matches and nonmatches.
     * @param wordsPathName path to file with testing sentences
     * @param tagsPathName path to file with testing tags
     * @throws Exception
     */
    public static void performance(String wordsPathName, String tagsPathName) throws Exception {

        List<List<String>> words = getTrainingLists(wordsPathName);
        List<List<String>> givenTestTags = getTrainingLists(tagsPathName);

        List<List<String>> predictedTestTags = new ArrayList<>();

        for(List<String> sentence:words){
            predictedTestTags.add(doViterbi(sentence));
        }

        int matches = 0;
        int nonmatches = 0;
        for (int i = 0; i < givenTestTags.size(); i++) {
            for (int j = 1; j < givenTestTags.get(i).size(); j++) {
                if (givenTestTags.get(i).get(j).equals(predictedTestTags.get(i).get(j))) {
                    matches += 1;
                } else {
                    nonmatches += 1;
                }
            }
        }
        System.out.println("Matches: " + matches+ "\nNon-Matches: " + nonmatches);
    }

    /**
     * A file-based test method to evaluate the performance on a pair of test files (corresponding lines with sentences and tags).
     * @param file1Path path of File 1
     * @param file2Path path of file 2
     * @throws Exception IOException
     */
    public static void testSimilarityOfTwoFiles(String file1Path, String file2Path) throws Exception {

        List<List<String>> givenTestTags1 = getTrainingLists(file1Path);
        List<List<String>> givenTestTags2 = getTrainingLists(file2Path);

        int matches = 0;
        int nonmatches = 0;
        for (int i = 0; i < givenTestTags1.size(); i++) {
            // if sentences have different lengths, checks the length till the smallest sentence
            // ends and then adds their size difference to nonmatches
            if(givenTestTags1.get(i).size()!=givenTestTags2.get(i).size()){
                nonmatches += Math.abs(givenTestTags1.get(i).size()-givenTestTags2.get(i).size());
            }
            int smallerOne = Math.min(givenTestTags1.get(i).size(),givenTestTags2.get(i).size());
            for (int j = 0; j < smallerOne; j++) {
                if (givenTestTags1.get(i).get(j).equals(givenTestTags2.get(i).get(j))) {
                    matches += 1;
                } else {
                    nonmatches += 1;
                }
            }
        }
        System.out.println("Matches: " + matches+ "\nNon-Matches: " + nonmatches);
    }

    /**
     * A console-based test method to give the tags from an input line.
     */
    public static void consoleDoViterbi(){
        Scanner scan = new Scanner(System.in);
        System.out.println("Type a sentence and view the tagged result:");
        String inp = scan.nextLine();
        if (inp.equals("q")) Runtime.getRuntime().exit(1);
        List<String> arr = new ArrayList<>(List.of(inp.split(" ")));
        arr.add(0,"#");
        System.out.println(arr);
        List<String> out = doViterbi(arr);
        System.out.println(out);
    }

    /**
     * Testing the method on simple hard-coded graphs and input strings
     * (e.g., from programming drill, along with others we make up).
     */
    public static void hardCodedTestFromRecitation(){
        //Hard-Coding:

        //---------------------------FOR TRANSITION MAP---------------------------\\

        Map<String, Map<String, Double>> transitionMapHardCoded = new HashMap<>();

        // # -> {...}
        Map<String, Double> t_forHashtag = new HashMap<>();
        t_forHashtag.put("n",5.00);
        t_forHashtag.put("np",2.00);
        transitionMapHardCoded.put("#", t_forHashtag);

        // np -> {...}
        Map<String, Double> t_forNP = new HashMap<>();
        t_forNP.put("v",2.00);
        transitionMapHardCoded.put("np", t_forNP);

        // cnj -> {...}
        Map<String, Double> t_forCNJ = new HashMap<>();
        t_forCNJ.put("n",1.00);
        t_forCNJ.put("np",1.00);
        t_forCNJ.put("v",1.00);
        transitionMapHardCoded.put("cnj", t_forCNJ);

        // n -> {...}
        Map<String, Double> t_forN = new HashMap<>();
        t_forN.put("v",6.00);
        t_forN.put("cnj",2.00);
        transitionMapHardCoded.put("n", t_forN);

        // v -> {...}
        Map<String, Double> t_forV = new HashMap<>();
        t_forV.put("n",6.00);
        t_forV.put("cnj",1.00);
        t_forV.put("np",2.00);
        transitionMapHardCoded.put("v", t_forV);

        tMap = normalizeAndLog(transitionMapHardCoded);

        //---------------------------FOR TRANSITION MAP---------------------------\\

        //----------------------------FOR EMISSION MAP----------------------------\\

        Map<String, Map<String, Double>> emissionMapHardCoded = new HashMap<>();

        // # -> {...}
        Map<String, Double> e_forHashtag = new HashMap<>();
        e_forHashtag.put("#",6.00);
        emissionMapHardCoded.put("#", e_forHashtag);

        // np -> {...}
        Map<String, Double> e_forNP = new HashMap<>();
        e_forNP.put("chase",5.00);
        emissionMapHardCoded.put("np", e_forNP);

        // cnj -> {...}
        Map<String, Double> e_forCNJ = new HashMap<>();
        e_forCNJ.put("and",3.00);
        emissionMapHardCoded.put("cnj", e_forCNJ);

        // n -> {...}
        Map<String, Double> e_forN = new HashMap<>();
        e_forN.put("cat",5.00);
        e_forN.put("dog",5.00);
        e_forN.put("watch",2.00);
        emissionMapHardCoded.put("n", e_forN);

        // v -> {...}
        Map<String, Double> e_forV = new HashMap<>();
        e_forV.put("watch",6.00);
        e_forV.put("get",1.00);
        e_forV.put("chase",2.00);
        emissionMapHardCoded.put("v", e_forV);

        eMap = normalizeAndLog(emissionMapHardCoded);

        //----------------------------FOR EMISSION MAP----------------------------\\

        String testSentenceFromRecitation = "dog watch chase chase watch";
        List<String> inp = new ArrayList<>(List.of(testSentenceFromRecitation.split(" ")));
        inp.add(0,"#");
        System.out.println(testSentenceFromRecitation);
        System.out.println(doViterbi(inp));

    }

    /**
     * Runs similarity testing on given dataset: Simple and Brown
     * @param trainX Training sentences
     * @param trainY Training tags
     * @param testX Testing sentences
     * @param testY Testing tags
     * @throws Exception IOException
     */
    public static void testDataset(String trainX, String trainY, String testX, String testY) throws Exception {
        List<List<String>> words = getTrainingLists(trainX);
        List<List<String>> tags = getTrainingLists(trainY);
        getTransitionAndEmissionMaps(words, tags);
        performance(testX, testY);
    }

    /**
     * Main function to test everything: Making model, trying out hard-coded method, console method, and testing performance.
     */
    public static void main(String[] args) throws Exception {

        partOfSpeechTagger model = new partOfSpeechTagger(); //makes model


        // Test the method on simple hard-coded graphs and input strings (e.g., from programming drill, along with others you make up).
        model.hardCodedTestFromRecitation();

        // Testing performance
        System.out.println("Simple Dataset Results: ");
        model.testDataset("simple-train-sentences.txt","simple-train-tags.txt","simple-test-sentences.txt","simple-test-tags.txt");
        System.out.println("Brown Dataset Results: ");
        model.testDataset("brown-train-sentences.txt","brown-train-tags.txt","brown-test-sentences.txt","brown-test-tags.txt");

        // Write a console-based test method to give the tags from an input line.
        while(true){
            model.consoleDoViterbi();
        }

    }
}
