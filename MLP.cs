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
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLP_TAKE2
{
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

        
        /*private int numberOfInputNeurons;
        private int numberOfOutputNeurons; //make it as a List with length of amount of hidden layers
        private int numberOfHiddenNeurons;*/

        public int numberOfInputNeurons;
        public int numberOfOutputNeurons; //make it as a List with length of amount of hidden layers
        public int numberOfHiddenNeurons;

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
        public void LoadData(string filename)
        {
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

        public void Train(int numberOfEpochs, double learningRate,double momentum,bool bias,bool shuffle)
        {
            //NA RAZIE JEST HARDCODOWANE BO CHCE MIEC TYLKO ROZPISANY ALGORYTM (numberOfSamples ofc)
            int numberOfSamples = 130;
            InitializeWeightsAndBiases(bias);

            for(int i =0; i<numberOfEpochs; i++)
            {
                double mse = 0d;
                if (shuffle)
                {
                    ShuffleData();
                }
                for(int j = 0; j < numberOfSamples; j++)
                {
                    getSample(j);
                    ForwardPropagation();
                    mse+=CalculateCostFunction();
                    BackwardPropagation();
                    ApplyNewWeights(learningRate,momentum);
                }
                //Console.WriteLine("Blad dla calej warstwy:" + mse);
                //Console.WriteLine("Sredni blad dla calej warstwy:" + mse/(double)numberOfSamples);
            }
        }

        private string convertMultipleListToString(double[,] list)
        {
            int rows = list.GetLength(0);
            int columns = list.GetLength(1);

            List<string> rowsAsStrings = new List<string>();

            for (int i = 0; i < rows; i++)
            {
                List<string> columnsAsStrings = new List<string>();

                for (int j = 0; j < columns; j++)
                {
                    columnsAsStrings.Add(list[i, j].ToString());
                }

                rowsAsStrings.Add(string.Join(" , ", columnsAsStrings));
            }

            string formattedString = string.Join("\n", rowsAsStrings);
            Console.WriteLine("multiple array string : " + formattedString + "\n\n");
            return formattedString;
        }

        private string convertSingleListToString(double[] list)
        {
            string formattedString = "";
            for (int i=0; i < list.Length; i++)
            {
                formattedString = string.Join(" , ", list[i].ToString());
            }
            Console.WriteLine("single array string : "+formattedString + "\n\n");
            return formattedString;
        }

        public void SaveNetworkToFile(String filename)
        {
            try
            {
                string mlpstring = numberOfInputNeurons.ToString() + " , " + numberOfHiddenNeurons.ToString() + " , " + numberOfOutputNeurons.ToString() + " endl ";
                mlpstring += convertSingleListToString(currentSample) + " endl ";
                mlpstring += convertSingleListToString(desiredOutput) + " endl ";
                mlpstring += convertMultipleListToString(hiddenLayerMatrix) + " endl ";
                mlpstring += convertMultipleListToString(outputLayerMatrix) + " endl ";
                mlpstring += convertMultipleListToString(hiddenLayerGradientMatrix) + " endl ";
                mlpstring += convertMultipleListToString(outputLayerGradientMatrix) + " endl ";
                mlpstring += convertMultipleListToString(hiddenLayerDeltaMatrix) + " endl ";
                mlpstring += convertMultipleListToString(outputLayerDeltaMatrix) + " endl ";
                mlpstring += convertSingleListToString(hiddenLayerBiasGradient) + " endl ";
                mlpstring += convertSingleListToString(outputLayerBiasGradient) + " endl ";
                mlpstring += convertSingleListToString(hiddenLayerBiasDelta) + " endl ";
                mlpstring += convertSingleListToString(outputLayerBiasDelta) + " endl ";
                mlpstring += convertSingleListToString(hiddenLayerBias) + " endl ";
                mlpstring += convertSingleListToString(outputLayerBias) + " endl ";
                mlpstring += convertSingleListToString(weighedSumsHiddenLayer) + " endl ";
                mlpstring += convertSingleListToString(hiddenLayerOutputs) + " endl ";
                mlpstring += convertSingleListToString(weighedSumsOutputLayer) + " endl ";
                mlpstring += convertSingleListToString(outputLayerOutputs) + " endl ";

                File.WriteAllText(filename, mlpstring);
                Console.WriteLine("\nmlpstring\n" + mlpstring + "\n\n");
                Console.WriteLine("Class saved to file: " + filename);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error saving class to file: " + ex.Message);
            }
        }
        
        private double[,] convertStringToMultipleArray(string line)
        {
            Console.WriteLine("multiple: ");
            string[] rowStrings = line.Split(new[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);
            int rows = rowStrings.Length;
            int columns = rowStrings[0].Split(new[] { " , " }, StringSplitOptions.RemoveEmptyEntries).Length;

            Console.WriteLine(rows + " _ " + columns + "\n");
            Console.WriteLine(line + "\n\n");
            double[,] array = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                string[] columnStrings = rowStrings[i].Split(new[] { " , " }, StringSplitOptions.RemoveEmptyEntries);

                for (int j = 0; j < columns; j++)
                {
                    if (double.TryParse(columnStrings[j], out double value))
                    {
                        array[i, j] = value;
                    }
                    else
                    {
                        Console.WriteLine("error in converting at[" + i + ","+j+"]: " + columnStrings[j]);
                    }
                }
            }
            return array;
        }

        private double[] convertStringToSingleArray(string line)
        {
            string[] valueStrings = line.Split(new[] { " , " }, StringSplitOptions.None);
            double[] list = new double[valueStrings.Length];

            Console.WriteLine("single: " + valueStrings.Length + "\n");
            Console.WriteLine(line + "\n\n");
            for (int i = 0; i < valueStrings.Length; i++)
            {
                if (double.TryParse(valueStrings[i], out double value))
                {
                    list[i] = value;
                }
                else
                {
                    Console.WriteLine("error in converting at ["+ i+ "]: " + valueStrings[i]);
                }
                Console.WriteLine("list[" + i + "]: " + list[i]);
            }
            return list;
        }

        public static MLP ReadNetworkFromFile(String filename, MLP obj)
        {
            try
            {
                string fileContent = File.ReadAllText(filename);
                
                string[] lines = fileContent.Split(new[] { " endl " }, StringSplitOptions.RemoveEmptyEntries);
                string[] firstline = lines[0].Split(new[] {" , "}, StringSplitOptions.RemoveEmptyEntries);
                int.TryParse(firstline[0], out int inputNeurons);
                int.TryParse(firstline[1], out int hiddenNeurons);
                int.TryParse(firstline[2], out int outputNeurons);
                //MLP obj = new MLP(inputNeurons, hiddenNeurons, outputNeurons);
                obj.numberOfInputNeurons = inputNeurons;
                obj.numberOfHiddenNeurons = hiddenNeurons;
                obj.numberOfOutputNeurons = outputNeurons;
                Console.WriteLine("inputs: "+obj.numberOfInputNeurons+", hidden: "+obj.numberOfHiddenNeurons+", output: "+obj.numberOfOutputNeurons+"\n\n");
                obj.setArraysFromFile(lines);


                Console.WriteLine("Class loaded from file: " + filename);
                return obj;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error reading class from file: " + ex.Message);
                return null;
            }
        }

        public void setArraysFromFile(string[] stringArrays)
        {
            currentSample = convertStringToSingleArray(stringArrays[1]);
            desiredOutput = convertStringToSingleArray(stringArrays[2]);

            hiddenLayerMatrix = convertStringToMultipleArray(stringArrays[3]);
            outputLayerMatrix = convertStringToMultipleArray(stringArrays[4]);

            hiddenLayerGradientMatrix = convertStringToMultipleArray(stringArrays[5]);
            outputLayerGradientMatrix = convertStringToMultipleArray(stringArrays[6]);

            hiddenLayerDeltaMatrix = convertStringToMultipleArray(stringArrays[7]);
            outputLayerDeltaMatrix = convertStringToMultipleArray(stringArrays[8]);

            hiddenLayerBiasGradient = convertStringToSingleArray(stringArrays[9]);
            outputLayerBiasGradient = convertStringToSingleArray(stringArrays[10]);

            hiddenLayerBiasDelta = convertStringToSingleArray(stringArrays[11]);
            outputLayerBiasDelta = convertStringToSingleArray(stringArrays[12]);


            hiddenLayerBias = convertStringToSingleArray(stringArrays[13]);
            outputLayerBias = convertStringToSingleArray(stringArrays[14]);
            
            weighedSumsHiddenLayer = convertStringToSingleArray(stringArrays[15]);
            hiddenLayerOutputs = convertStringToSingleArray(stringArrays[16]);

            weighedSumsOutputLayer = convertStringToSingleArray(stringArrays[17]);
            outputLayerOutputs = convertStringToSingleArray(stringArrays[18]);
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

        public void Test(int firstSampleIndex, int lastSampleIndex)
        {
            Console.WriteLine("The result:");
            int correctlyClassified = 0;
            int incorrectlyClassified = 0;
            for(int i = firstSampleIndex; i < lastSampleIndex; i++)
            {
                getSample(i);
                ForwardPropagation();
                double cost = CalculateCostFunction();
                if (cost < 0.005)
                {
                    correctlyClassified++;
                }
                else
                {
                    incorrectlyClassified++;
                }
                /*
                Console.Write(outputLayerOutputs[0]);
                Console.Write(outputLayerOutputs[1]);
                Console.Write(outputLayerOutputs[2]);
                */
            }
            Console.WriteLine("Correctly classified:" + correctlyClassified);
            Console.WriteLine("Incorrectly classified:" + incorrectlyClassified);
        }


        private void getSample(int numberOfSample)
        {
            for(int i = 0; i < numberOfInputNeurons; i++)
            {
                try
                {
                    Console.WriteLine("data["+numberOfSample+"] length: " + data.ElementAt(i).Count + " i: "+i);
                    currentSample[i] = data.ElementAt(numberOfSample).ElementAt(i);

                } catch (Exception e) { Console.WriteLine("number of samples error: " + e); }
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

        private void InitializeWeightsAndBiases(bool bias)
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
                // src: https://www.quora.com/How-is-bias-updated-in-neural-network
                // PSA: Yes I am really desperate so I'm gonna take answers from quora.
                // Take this with a pinch of salt, I am not sure if I'm doing it the right way.
                outputLayerBiasGradient[i] = 1.0 * SigmoidDerivative(weighedSumsOutputLayer[i])
                        * (outputLayerOutputs[i] - desiredOutput[i]);
            }


            // HIDDEN LAYER BACKWARD PROPAGATION

            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    // I added this not so long ago, I didn't clear out the matrix.
                    hiddenLayerGradientMatrix[i, j] = 0;
                    for (int k = 0; k < numberOfOutputNeurons; k++)
                    {
                        // formula is on paper, point 9
                        // I swear to god if this works i'm going to apply to openai
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
                hiddenLayerBiasGradient[i] *= 1.0 / numberOfOutputNeurons;
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
