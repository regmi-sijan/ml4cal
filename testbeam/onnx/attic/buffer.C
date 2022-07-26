    // Original place for inputTensorValues.assign
    inputTensorValues.assign(testArray);
    // inputTensorValues.assign({1554.0, 1558.0, 1555.0,  1564.0, 1558.0, 1555.0, 1556.0, 1554.0, 1750.0, 2284.0, 2424.0, 2116.0, 1838.0, 1713.0, 1649.0, 1613.0, 1601.0, 1589.0, 1583.0, 1578.0, 1572.0, 1574.0, 1573.0, 1569.0, 1567.0, 1562.0, 1563.0, 1560.0, 1561.0, 1557.0, 1557.0});

    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));
        


    // cout<<testArray<<endl;

    // see comment above
    // inputTensorValues.assign(testArray.data);

    if (verbose) {
        std::cout <<inputTensorValues<< std::endl;
    }

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);            

    std::cout << outputTensorValues << std::endl;