using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace MLP_TAKE2
{
    [Serializable]
    internal class MLP
    {
        private double[] currentSample;

        // Desired inputs in format:
        // First index is the index of output neuron
        // Second index is the index of sample
        private double[] desiredOutput;

        // Hidden layer and output layer matrices are in this form:
        // First index is the index of neuron
        // second index is the index of weights
        private double[,] hiddenLayerMatrix;
        private double[,] outputLayerMatrix;

        private double[,] hiddenLayerGradientMatrix;
        private double[,] outputLayerGradientMatrix;

        private double[,] hiddenLayerDeltaMatrix;
        private double[,] outputLayerDeltaMatrix;

        private double[] hiddenLayerBiasGradient;
        private double[] outputLayerBiasGradient;

        private double[] hiddenLayerBiasDelta;
        private double[] outputLayerBiasDelta;


        private double[] hiddenLayerBias;
        private double[] outputLayerBias;

        
        private int numberOfInputNeurons;
        private int numberOfOutputNeurons; //make it as a List with length of amount of hidden layers
        private int numberOfHiddenNeurons;


        private double[] weighedSumsHiddenLayer;
        private double[] hiddenLayerOutputs;

        private double[] weighedSumsOutputLayer;
        private double[] outputLayerOutputs;



        private List<List<double>> data;
        private List<List<double>> desiredOutputs;

        public MLP(int numberOfInputNeurons, int numberOfHiddenNeurons, int numberOfOutputs) //(list<int> layers=[inputammout, hiddenammount1, hiddenammount2, ..., hiddenammountn, outputammount], int bias)
        {
            this.numberOfInputNeurons = numberOfInputNeurons; //layers[0]
            this.numberOfHiddenNeurons = numberOfHiddenNeurons;//for i=1; i<layers.length-1; i++
            this.numberOfOutputNeurons = numberOfOutputs; //layers[-1]

            desiredOutput = new double[numberOfOutputs];
            currentSample = new double[numberOfInputNeurons];
            hiddenLayerMatrix = new double[numberOfHiddenNeurons, numberOfInputNeurons];
            outputLayerMatrix = new double[numberOfOutputs, numberOfHiddenNeurons];
            hiddenLayerBias = new double[numberOfHiddenNeurons];
            outputLayerBias = new double[numberOfOutputs];

            hiddenLayerBiasGradient = new double[numberOfHiddenNeurons];
            outputLayerBiasGradient = new double[numberOfOutputNeurons];


            weighedSumsHiddenLayer = new double[numberOfHiddenNeurons];
            hiddenLayerOutputs = new double[numberOfHiddenNeurons];
            hiddenLayerGradientMatrix = new double[numberOfHiddenNeurons, numberOfInputNeurons];
            hiddenLayerDeltaMatrix = new double[numberOfHiddenNeurons, numberOfInputNeurons];
            hiddenLayerBiasDelta = new double[numberOfHiddenNeurons];
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                hiddenLayerBiasGradient[i] = 0;
                weighedSumsHiddenLayer[i] = 0;
                hiddenLayerOutputs[i] = 0;
                hiddenLayerBiasDelta[i] = 0;
                for(int j = 0; j< numberOfInputNeurons; j++)
                {
                    hiddenLayerGradientMatrix[i,j] = 0;
                    hiddenLayerDeltaMatrix[i,j] = 0;
                }
            }
            
            outputLayerOutputs = new double[numberOfOutputNeurons];
            weighedSumsOutputLayer = new double[numberOfOutputNeurons]; //tu bylo numberOfHiddenNeurons, ale to byl razcej blad (nie krytyczny, program dzialal, ale duzy blad)
            outputLayerGradientMatrix = new double[numberOfOutputNeurons, numberOfHiddenNeurons];
            outputLayerDeltaMatrix = new double[numberOfOutputNeurons, numberOfHiddenNeurons];
            outputLayerBiasDelta = new double[numberOfOutputNeurons];
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                outputLayerBiasGradient[i] = 0;
                weighedSumsOutputLayer[i] = 0;
                outputLayerOutputs[i] = 0;
                outputLayerBiasDelta[i] = 0;
                for(int j = 0; j< numberOfHiddenNeurons; j++)
                {
                    outputLayerGradientMatrix[i,j] = 0;
                    outputLayerDeltaMatrix[i, j] = 0;
                }
            }


            data = new List<List<double>>();
            desiredOutputs = new List<List<double>>();
        }
        public void LoadIrisData(string filename)
        {
            data.Clear();
            string[] lines = File.ReadAllLines(filename);
            foreach (string line in lines)
            {
                string[] columns = line.Split(',');
                List<double> rowData = new List<double>(columns.Length);

                // Parse the first four columns as double values
                for (int i = 0; i < 4; i++)
                {
                    rowData.Add(ConvertToDouble(columns[i]));
                }
                if (columns[4] == "Iris-setosa")
                {
                    List<double> output = new List<double>();
                    rowData.Add(1);
                    rowData.Add(0);
                    rowData.Add(0);
                }
                else if (columns[4] == "Iris-versicolor")
                {
                    List<double> output = new List<double>();
                    rowData.Add(0);
                    rowData.Add(1);
                    rowData.Add(0);
                    desiredOutputs.Add(output);
                }
                else if (columns[4] == "Iris-virginica")
                {
                    List<double> output = new List<double>();
                    rowData.Add(0);
                    rowData.Add(0);
                    rowData.Add(1);
                    desiredOutputs.Add(output);
                }
                data.Add(rowData);
            }
        }
        public void LoadAutoEncoderData(string filename)
        {
            data.Clear();
            string[] lines = File.ReadAllLines(filename);
            foreach (string line in lines)
            {
                string[] columns = line.Split(',');
                List<double> rowData = new List<double>(columns.Length);

                // Parse the first four columns as double values
                for (int i = 0; i < 8; i++)
                {
                    rowData.Add(ConvertToDouble(columns[i]));
                }
                data.Add(rowData);
            }
        }


        public static double ConvertToDouble(string input)
        {
            double result = 0.0;
            double sign = 1.0;
            double fraction = 0.0;
            bool hasFraction = false;
            int decimalPlaces = 1;
            int currentIndex = 0;

            // Check if the input string is empty
            if (string.IsNullOrEmpty(input))
            {
                throw new ArgumentException("Invalid input. The string is empty.");
            }

            // Check for a sign character (+ or -) at the beginning
            if (input[currentIndex] == '-')
            {
                sign = -1.0;
                currentIndex++;
            }
            else if (input[currentIndex] == '+')
            {
                currentIndex++;
            }

            // Iterate over the remaining characters
            while (currentIndex < input.Length)
            {
                char currentChar = input[currentIndex];

                // Check for a decimal point
                if (currentChar == '.')
                {
                    if (hasFraction)
                    {
                        throw new ArgumentException("Invalid input. The string contains multiple decimal points.");
                    }

                    hasFraction = true;
                    currentIndex++;
                    continue;
                }

                // Convert the character to a digit
                int digit = currentChar - '0';

                // Check if the character is a valid digit
                if (digit < 0 || digit > 9)
                {
                    throw new ArgumentException("Invalid input. The string contains non-numeric characters.");
                }

                // Update the result based on the current digit
                if (hasFraction)
                {
                    fraction = fraction * 10.0 + digit;
                    decimalPlaces *= 10;
                }
                else
                {
                    result = result * 10.0 + digit;
                }

                currentIndex++;
            }

            // Apply the sign and fraction (if any) to the final result
            result = sign * (result + fraction / decimalPlaces);

            return result;
        }

        public void Train(int numberOfEpochs, double learningRate,double momentum,bool bias,bool shuffle, double minError, bool loadedFromFile)
        {
            // UWAGA, HARDCODOWANY NUMBER OF SAMPLES
            //int numberOfSamples = 100;
            if (!loadedFromFile)
            {
                InitializeWeightsAndBiases(bias);
            }
            string toFile = "";
            for(int i =0; i<numberOfEpochs; i++)
            {
                double mse = 0d;
                if (shuffle)
                {
                    ShuffleData();
                }
                for(int j = 0; j < data.Count; j++)
                {
                    getSample(j);
                    ForwardPropagation();
                    mse+=CalculateCostFunction();
                    BackwardPropagation();
                    ApplyNewWeights(learningRate,momentum);
                    if (!bias)
                    {
                        SetBiasesToZero();
                    }
                }
                if (mse<minError)
                {
                    Console.WriteLine("Reached minerror");
                    break;
                }
                if (i % 10 == 0) toFile += i + " " + mse + "\n";
            }
            saveStringToFile("../../../Stats/globalError.data", toFile);
            
        }

        private void saveStringToFile(string filePath,string text)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.Write(text);
            }
        }

        public void SaveNetworkToFile(String filename)
        {
            try
            {
                // Create a FileStream to write the serialized data
                FileStream fileStream = new FileStream(filename, FileMode.Create);

                // Create a BinaryFormatter to perform the serialization
                BinaryFormatter formatter = new BinaryFormatter();

                // Serialize the object to the file
                formatter.Serialize(fileStream, this);

                // Close the file stream
                fileStream.Close();

                Console.WriteLine("Class saved to file: " + filename);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error saving class to file: " + ex.Message);
            }
        }
        
        public static MLP ReadNetworkFromFile(String filename)
        {
            try
            {
                // Create a FileStream to read the serialized data
                FileStream fileStream = new FileStream(filename, FileMode.Open);

                // Create a BinaryFormatter to perform the deserialization
                BinaryFormatter formatter = new BinaryFormatter();

                // Deserialize the object from the file
                MLP obj = (MLP)formatter.Deserialize(fileStream);

                // Close the file stream
                fileStream.Close();

                Console.WriteLine("Class loaded from file: " + filename);

                return obj;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error reading class from file: " + ex.Message);
                return null;
            }
        }

        private void ShuffleData()
        {
            Random random = new Random();

            // Start from the end and swap each element with a randomly selected one
            for (int i = data.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                List<double> temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }

        private void CollectTestData(int sampleIndex)
        {
            string data = "";
            double[] outputErrors = new double[numberOfOutputNeurons];
            double totalError = 0;
            string sample = "\nWzorzec wejsciowy: ";
            string desired = "\nPrzewidywane wyjscie: ";
            string forwarded = "\nOtrzymane wyjscie: ";
            for (int i = 0; i < numberOfInputNeurons; i++)
            {
                sample += currentSample[i];
            }
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                desired += desiredOutput[i];
                forwarded += outputLayerOutputs[i] + " ";
                outputErrors[i] = outputLayerOutputs[i] - desiredOutput[i];
                totalError += outputErrors[i];
            }
            string outputsOfHiddenLayer = "\nWyjscia warstwy ukrytej: \n";
            for(int i = 0; i< numberOfHiddenNeurons; i++)
            {
                outputsOfHiddenLayer += hiddenLayerOutputs[i] + " ";
            }

            string weightMatrixOutput = "\nMacierz wag warstwy wyjsciowej: \n ";
            for(int i = 0; i< numberOfOutputNeurons; i++)
            {
                for(int j = 0; j < numberOfHiddenNeurons; j++)
                {
                    weightMatrixOutput += outputLayerMatrix[i,j] + " ";
                }
                weightMatrixOutput += "\n";
            }
            string weightMatrixHidden = "\nMacierz wag warstwy ukrytej: \n ";
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    weightMatrixHidden += hiddenLayerMatrix[i, j] + " ";
                }
                weightMatrixHidden += "\n";
            }
            data += sample + desired + forwarded + outputsOfHiddenLayer + weightMatrixOutput + weightMatrixHidden;
            string Path = "../../../Stats/Testing" + sampleIndex + ".data";
            saveStringToFile(Path, data);
        }
        public void TestIrises()
        {
            Console.WriteLine("The result:");
            int correctlyClassified = 0;
            int incorrectlyClassified = 0;
            List<List<int>> confusionMatrix = new List<List<int>>();

            //initialize matrix with zeroes
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                List<int> row = new List<int>();

                for (int j = 0; j < numberOfOutputNeurons; j++)
                {
                    row.Add(0);
                }

                confusionMatrix.Add(row);
            }

            // iterujemy po wszystkich testowych
            for (int i = 0; i < data.Count; i++)
            {
                getSample(i);
                ForwardPropagation();
                CollectTestData(i);
                int maxIndex = 0;
                int maxExpectedIndex = 0;
                // -1 because i want the value to be smaller than any element in the array
                double maxValue = -1;
                double maxExpectedValue = -1;
                for (int j = 0; j < numberOfOutputNeurons; j++)
                {
                    if (outputLayerOutputs[j] > maxValue)
                    {
                        maxIndex = j;
                        maxValue = outputLayerOutputs[j];
                    }
                    if (desiredOutput[j] > maxExpectedValue)
                    {
                        maxExpectedIndex = j;
                        maxExpectedValue = outputLayerOutputs[j];
                    }
                }
                confusionMatrix[maxExpectedIndex][maxIndex]++;
            }
            // CALCULATING TP, TN, FP, FN
            // https://12ft.io/proxy?q=https%3A%2F%2Ftowardsdatascience.com%2Fconfusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
            // SOURCE UPWARD
            string result = "";
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                result += "CLASS:" + i + "\n";
                int tp = confusionMatrix[i][i];
                int tn = 0;
                int fp = 0;
                int fn = 0;
                // TRUE NEGATIVE
                for(int j = 0; j < numberOfOutputNeurons;j++)
                {
                    for (int k = 0; k < numberOfOutputNeurons; k++)
                    {
                        if(k != i && j != i)
                        {
                            tn += confusionMatrix[j][k];
                        }
                    }
                }
                //FALSE POSITIVE AND FALSE NEGATIVE
                for(int j = 0; j<numberOfOutputNeurons; j++)
                {
                    if (j != i)
                    {
                        fp+=confusionMatrix[j][i];
                        fn+=confusionMatrix[i][j];
                    }
                }

                result += "TP: " + tp + " TN: " + tn + " FP: " + fp + " FN: " + fn;
                result += "\n";
                // SOURCE FOR METRICS:
                // https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/

                double precision = (double)tp /(double)(tp + fp);
                double recall = tp / (tp + fn);
                double fmeasure = (2 * precision * recall) / (precision + recall);
                result += "Precision: " + precision + " Recall: " + recall + " F-measure: " + fmeasure + "\n\n";

                // not sure about that but "na chlopski rozum" it works
                correctlyClassified += tp;
                incorrectlyClassified += fp;
            }
            result+= "Correctly classified:" + correctlyClassified + "\n";
            result += "Incorrectly classified:" + incorrectlyClassified + "\n";
            saveStringToFile("../../../Stats/confusionMatrix.data", result);
        }

        private void getSample(int numberOfSample)
        {
            for(int i = 0; i < numberOfInputNeurons; i++)
            {
                currentSample[i] = data.ElementAt(numberOfSample).ElementAt(i);
            }
            for(int i = numberOfInputNeurons; i< numberOfOutputNeurons+numberOfInputNeurons; i++)
            {
                desiredOutput[i-numberOfInputNeurons] = data.ElementAt(numberOfSample).ElementAt(i);
            }
        }
        private void ApplyNewWeights(double learningRate,double momentum)
        {
            //Applying for hidden layer
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    hiddenLayerMatrix[i, j] += learningRate * -1.0 * hiddenLayerGradientMatrix[i, j] 
                        + hiddenLayerDeltaMatrix[i,j]*momentum; 
                    // MOMENTUM
                    hiddenLayerDeltaMatrix[i,j] = learningRate * -1.0 * hiddenLayerGradientMatrix[i, j]
                        + hiddenLayerDeltaMatrix[i, j] * momentum;
                }
                hiddenLayerBias[i] += learningRate * -1.0 * hiddenLayerBiasGradient[i] + hiddenLayerBiasDelta[i] * momentum;
                hiddenLayerBiasDelta[i] = learningRate * -1.0 * hiddenLayerBiasGradient[i] + hiddenLayerBiasDelta[i] * momentum;
            }
            //Apply for output layer
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons; j++)
                {
                    outputLayerMatrix[i, j] += learningRate * -outputLayerGradientMatrix[i, j] + outputLayerDeltaMatrix[i,j]*momentum;
                    // MOMENTUM
                    outputLayerDeltaMatrix[i,j] = learningRate * -outputLayerGradientMatrix[i, j] + outputLayerDeltaMatrix[i, j] * momentum;
                }
                outputLayerBias[i] += learningRate * -1.0 * outputLayerBiasGradient[i] + outputLayerBiasDelta[i]*momentum;
                outputLayerBiasDelta[i] = learningRate * -1.0 * outputLayerBiasGradient[i] + outputLayerBiasDelta[i]*momentum;
            }
        }

        public void SetBiasesToZero()
        {
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                hiddenLayerBias[i] = 0d;
            }
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                outputLayerBias[i] = 0d;
            }
        }
        public void InitializeWeightsAndBiases(bool bias)
        {
            Random random = new Random();

            // INITIALIZATION FOR HIDDEN LAYER
            for ( int i = 0; i < numberOfHiddenNeurons ; i++)
            {
                for(int j = 0; j < numberOfInputNeurons ; j++)
                { 
                    hiddenLayerMatrix[i,j] = random.NextDouble() * 2 - 1; // randomizing a number in range [-1,1]
                }
                if (bias)
                {
                    hiddenLayerBias[i] = random.NextDouble() * 2 - 1;   // same here
                }
                else
                {
                    hiddenLayerBias[i] = 0d;
                }
            }
            // INITIALIZATION FOR OUTPUT LAYER
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons; j++)
                {
                    outputLayerMatrix[i, j] = random.NextDouble() * 2 - 1; // randomizing a number in range [-1,1]
                }
                if (bias)
                {
                    outputLayerBias[i] = random.NextDouble() * 2 - 1; // same here
                }
                else
                {
                    outputLayerBias[i] = 0;
                }
            }

        }
        private void ForwardPropagation()
        {
            // FORWARD PROPGATION FOR HIDDEN LAYER
            

            for(int i = 0; i<numberOfHiddenNeurons; i++)
            {
                weighedSumsHiddenLayer[i] = 0;
                for(int j = 0; j < numberOfInputNeurons; j++)
                {
                    weighedSumsHiddenLayer[i] += currentSample[j] * hiddenLayerMatrix[i,j]; 
                }
                weighedSumsHiddenLayer[i] += hiddenLayerBias[i];
            }
            

            for(int i = 0; i < numberOfHiddenNeurons; i++)
            {
                hiddenLayerOutputs[i] = Sigmoid(weighedSumsHiddenLayer[i]);
            }

            // FORWARD PROPGATION FOR OUTPUT LAYER
            
            for (int i = 0; i<numberOfOutputNeurons; i++)
            {
                weighedSumsOutputLayer[i] = 0;
                for(int j = 0; j< numberOfHiddenNeurons; j++)
                {
                    weighedSumsOutputLayer[i] += hiddenLayerOutputs[j] * outputLayerMatrix[i, j];
                }
                weighedSumsOutputLayer[i] += outputLayerBias[i];
            }

            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                outputLayerOutputs[i] = Sigmoid(weighedSumsOutputLayer[i]);
            }
        }

        private double CalculateCostFunction()
        {
            double costFunction = 0;
            for(int i = 0; i < numberOfOutputNeurons ; i++)
            {
                costFunction += 0.5 * Math.Pow((outputLayerOutputs[i] - desiredOutput[i]),2);
            }
            costFunction /= numberOfOutputNeurons;
            return costFunction;
        }

        private void BackwardPropagation()
        {
            // OUTPUT LAYER BACKWARD PROPAGATION

            for(int i = 0; i < numberOfOutputNeurons ; i++)
            {
                for(int j = 0; j<numberOfHiddenNeurons ; j++)
                {
                    // formula is on paper, point 8
                    outputLayerGradientMatrix[i, j] = hiddenLayerOutputs[j] * SigmoidDerivative(weighedSumsOutputLayer[i])
                        * (outputLayerOutputs[i] - desiredOutput[i]);
                }
                // The weight of the bias term in a layer is updated in the same fashion as all the other weights are.
                // What makes it different is that it is independent of output from previous layers.
                // The weight for the bias term in a layer is always fed an input of 1.
                outputLayerBiasGradient[i] = 1.0 * SigmoidDerivative(weighedSumsOutputLayer[i])
                        * (outputLayerOutputs[i] - desiredOutput[i]);
            }


            // HIDDEN LAYER BACKWARD PROPAGATION

            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    hiddenLayerGradientMatrix[i, j] = 0;
                    for (int k = 0; k < numberOfOutputNeurons; k++)
                    {
                        // formula is on paper, point 9
                        hiddenLayerGradientMatrix[i, j] += (outputLayerOutputs[k] - desiredOutput[k]) * SigmoidDerivative(weighedSumsOutputLayer[k]) * outputLayerMatrix[k, i] *
                            SigmoidDerivative(weighedSumsHiddenLayer[i]) * currentSample[j];
                    }
                    double result = hiddenLayerGradientMatrix[i, j] * (1.0 / numberOfOutputNeurons);
                    hiddenLayerGradientMatrix[i, j] = result;
                }
                

            }
            // HERE WE CALCULATE BIAS DERIVATIVE FOR HIDDEN LAYER
            // Take this with a pinch of salt, I am not sure if I'm doing it the right way.
            for(int i = 0; i < numberOfHiddenNeurons; i++)
            {
                hiddenLayerBiasGradient[i] = 0;
                for(int j = 0; j< numberOfOutputNeurons; j++)
                {
                    hiddenLayerBiasGradient[i] += (outputLayerOutputs[j] - desiredOutput[j]) * SigmoidDerivative(weighedSumsOutputLayer[j]) * SigmoidDerivative(weighedSumsHiddenLayer[i]) *
                        outputLayerMatrix[j, i];
                }
                //hiddenLayerBiasGradient[i] *= 1.0 / numberOfOutputNeurons;
            }
        }


        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return Sigmoid(x)*(1-Sigmoid(x));
        }
    }
}
