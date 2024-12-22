# Define the batch size
$batchSize = 3  # Number of files per commit

# Get the list of files staged for commit
$files = git diff --cached --name-only

# Create batches of files
$batchedFiles = @()
$currentBatch = @()

foreach ($file in $files) {
    $currentBatch += $file
    if ($currentBatch.Count -ge $batchSize) {
        $batchedFiles += ,@($currentBatch)
        $currentBatch = @()  # Reset current batch
    }
}

# Add any remaining files in the last batch
if ($currentBatch.Count -gt 0) {
    $batchedFiles += ,@($currentBatch)
}

# Commit and push each batch
foreach ($batch in $batchedFiles) {
    # Add files to the staging area
    git add $batch

    # Commit the batch
    git commit -m "Batch commit of files"

    # Try to push the batch, handle failure
    $pushSuccess = $false
    try {
        git push origin main
        $pushSuccess = $true
    } catch {
        Write-Host "Push failed for batch $($batch -join ', ')"
    }

    # If push fails, skip this batch and move on to the next
    if ($pushSuccess) {
        Write-Host "Push successful for batch $($batch -join ', ')"
    } else {
        Write-Host "Skipping failed batch $($batch -join ', ')"
    }

    # Optional: Sleep for a few seconds between pushes
    Start-Sleep -Seconds 5
}

Write-Host "Batching files into groups of $batchSize..."
Write-Host "Found $($files.Count) files staged for commit."
Write-Host "Created $($batchedFiles.Count) batches of files."

foreach ($batch in $batchedFiles) {
    Write-Host "Processing batch $($batch.Count) files: $($batch -join ', ')"
    # Add files to the staging area
    git add $batch
    Write-Host "Added files to staging area."

    # Commit the batch
    git commit -m "Batch commit of files"
    Write-Host "Committed batch."

    # Try to push the batch, handle failure
    $pushSuccess = $false
    try {
        git push origin main
        $pushSuccess = $true
        Write-Host "Pushed batch to origin/main."
    } catch {
        Write-Host "Push failed for batch $($batch -join ', ')"
        Write-Host "Error: $($Error[0].Message)"
    }

    # If push fails, skip this batch and move on to the next
    if ($pushSuccess) {
        Write-Host "Push successful for batch $($batch -join ', ')"
    } else {
        Write-Host "Skipping failed batch $($batch -join ', ')"
    }

    # Optional: Sleep for a few seconds between pushes
    Write-Host "Sleeping for 5 seconds..."
    Start-Sleep -Seconds 5
}