
using MLP_TAKE2;
using System.Runtime.Serialization.Formatters.Binary;
/*
MLP mlp;
bool bias = false;
bool fromFile = false;
String mode = " ";

while (mode != "1" && mode != "2")
{
    Console.WriteLine("Wybierz tryb programu:\n1. Nauka\n2. Testowanie");
    mode = Console.ReadLine();
}

String dataset = " ";

while (dataset != "1" && dataset != "2")
{
    Console.WriteLine("Wybierz zestaw danych:\n1. Irysy\n2. Autoencoder");
    dataset = Console.ReadLine();
}

String usebias = " ";
while (usebias != "t" &&  usebias != "n"){
    Console.WriteLine("Czy ma zostać uwzględniony bias? [t/n]: ");
    usebias = Console.ReadLine();
}

if(usebias=="t")
{
    bias = true;
}

String fromfile = " ";
while (fromfile != "t" && fromfile != "n")
{
    Console.WriteLine("Czy chcesz wczytać sieć z pliku? [t/n]: ");
    fromfile = Console.ReadLine();
}
if (fromfile == "t")
{
    fromFile = true;
}
if (fromFile)
{
    String filepath = "";

    while (filepath == "")
    {
        Console.WriteLine("Podaj ścieżkę do pliku sieci: "); //add write and read network file, write one of the given amounts in instructions during training
        filepath = Console.ReadLine();
    }
    mlp = ReadNetworkFromFile(filepath);
} else
{
    List<int> layers = new List<int>();
    if (dataset == "1")
    {
        Console.WriteLine("Podaj ilość warstw ukrytych: ");
        int hiddenLayerNumber = Convert.ToInt32(Console.ReadLine());
        
        layers.Add(4);
        for (int i = 0;  i < hiddenLayerNumber; i++)
        {
            Console.WriteLine("Podaj ilość neuronów w warstwie ukrytej:");
            layers.Add(Convert.ToInt32(Console.ReadLine()));
        }
        layers.Add(3);
    } else
    {
        layers.Add(4);
        layers.Add(2);
        layers.Add(4);
    }
    //mlp = new MLP(layers, bias); //after changing for structure
}
mlp = new MLP(4, 4, 3, true); //for now

int epochs = 0;
if(mode == "1")
{
    while (epochs == 0)
    {
        Console.WriteLine("Podaj ilość iteracji w treningu: ");
        epochs = Convert.ToInt32(Console.ReadLine());
    }
    String saveToFilename = "";
    while(saveToFilename == "")
    {
        Console.WriteLine("Podaj nazwę pliku do zapisu sieci:");
        saveToFilename = Console.ReadLine();
    }
    
    mlp.Train(epochs, 0.1, 0.9);
}



MLP ReadNetworkFromFile(String filename)
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
*/
//mode == "1"  -> training
//mode == "2"  -> test 

//**************************************************************
List<int> neuronsInEachLayer = new List<int>();
neuronsInEachLayer.Add(4);
neuronsInEachLayer.Add(4);

MLP mlp = new MLP(4, 2, neuronsInEachLayer, 3);

mlp.LoadData("../../../training.data");

Console.WriteLine("TRAINING:");
mlp.Train(200,0.15,0.9,true,true);
Console.WriteLine("FINISHED TRAINING:");
Console.WriteLine("Testing learning samples:");
mlp.Test(0, 130);

//GC.Collect();
//MLP mlp2 = MLP.ReadNetworkFromFile("../../../test");

//mlp2.LoadData("../../../testing.data");
Console.WriteLine("Tests:");
mlp.Test(0,20);
