# PowerShell script to create full List2Cart project structure
$root = "list2cart"

# Function to create directories
function New-Dirs {
    param([string[]]$dirs)
    foreach ($dir in $dirs) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }
}

# Function to create empty files
function New-Files {
    param([string[]]$files)
    foreach ($file in $files) {
        New-Item -Path $file -ItemType File -Force | Out-Null
    }
}

# === Directories ===
$dirs = @(
    "$root\frontend\src\components\ui",
    "$root\frontend\src\components\layout",
    "$root\frontend\src\components\ai",
    "$root\frontend\src\components\cart",
    "$root\frontend\src\components\product",
    "$root\frontend\src\components\smart",
    "$root\frontend\src\layouts",
    "$root\frontend\src\pages\cart",
    "$root\frontend\src\pages\products",
    "$root\frontend\src\pages\smart",
    "$root\frontend\src\pages\api\ai",
    "$root\frontend\src\pages\api\cart",
    "$root\frontend\src/scripts",
    "$root\frontend\src/styles",
    "$root\frontend\src\utils",
    "$root\frontend\public\images\products",
    "$root\frontend\public\images\categories",
    "$root\frontend\public\images\ai-icons",
    "$root\frontend\public\images\cart-states",
    "$root\frontend\public\sounds",
    "$root\backend\src\routes\ai",
    "$root\backend\src\routes\cart",
    "$root\backend\src\controllers\ai",
    "$root\backend\src\services\ai",
    "$root\backend\src\database\schema",
    "$root\backend\src\database\migrations",
    "$root\backend\src\database\seeds",
    "$root\backend\src\models\ai",
    "$root\backend\src\middleware",
    "$root\backend\src\utils",
    "$root\backend\data\raw\kaggle-products",
    "$root\backend\data\raw\sample-lists",
    "$root\backend\data\processed",
    "$root\backend\data\images",
    "$root\backend\data\scripts",
    "$root\ai\engines",
    "$root\ai\models\ollama",
    "$root\ai\models\algorithms",
    "$root\ai\models\training",
    "$root\ai\prompts\cart-builder",
    "$root\ai\prompts\explanations",
    "$root\ai\prompts\templates",
    "$root\ai\utils",
    "$root\ai\config",
    "$root\ai\tests",
    "$root\shared\types\ai",
    "$root\shared\utils",
    "$root\scripts",
    "$root\docs"
)
New-Dirs $dirs

# === Files ===
$files = @(
    # Frontend UI
    "$root\frontend\src\components\ui\Button.astro",
    "$root\frontend\src\components\ui\Card.astro",
    "$root\frontend\src\components\ui\Modal.astro",
    "$root\frontend\src\components\ui\SearchBar.astro",
    "$root\frontend\src\components\ui\Loader.astro",
    "$root\frontend\src\components\ui\ProgressBar.astro",
    "$root\frontend\src\components\ui\Toast.astro",
    # Layout
    "$root\frontend\src\components\layout\Header.astro",
    "$root\frontend\src\components\layout\Footer.astro",
    "$root\frontend\src\components\layout\Navigation.astro",
    "$root\frontend\src\components\layout\MobileMenu.astro",
    # AI components
    "$root\frontend\src\components\ai\ListInput.astro",
    "$root\frontend\src\components\ai\CartBuilder.astro",
    "$root\frontend\src\components\ai\BuildProgress.astro",
    "$root\frontend\src\components\ai\SmartSuggestions.astro",
    "$root\frontend\src\components\ai\QuantityEstimator.astro",
    "$root\frontend\src\components\ai\ContextAnalyzer.astro",
    "$root\frontend\src\components\ai\CartPreview.astro",
    "$root\frontend\src\components\ai\AIInsights.astro",
    # Cart components
    "$root\frontend\src\components\cart\InstantCart.astro",
    "$root\frontend\src\components\cart\CartItem.astro",
    "$root\frontend\src\components\cart\CartSummary.astro",
    "$root\frontend\src\components\cart\QuickModify.astro",
    "$root\frontend\src\components\cart\CheckoutButton.astro",
    "$root\frontend\src\components\cart\CartOptimizer.astro",
    # Product components
    "$root\frontend\src\components\product\ProductCard.astro",
    "$root\frontend\src\components\product\ProductGrid.astro",
    "$root\frontend\src\components\product\ProductDetails.astro",
    "$root\frontend\src\components\product\CategoryBrowser.astro",
    "$root\frontend\src\components\product\QuickAdd.astro",
    "$root\frontend\src\components\product\AlternativeProducts.astro",
    # Smart
    "$root\frontend\src\components\smart\SmartSearch.astro",
    "$root\frontend\src\components\smart\PersonalizedDeals.astro",
    "$root\frontend\src\components\smart\ShoppingHistory.astro",
    "$root\frontend\src\components\smart\SmartFilters.astro",
    # Layouts
    "$root\frontend\src\layouts\Layout.astro",
    "$root\frontend\src\layouts\AILayout.astro",
    "$root\frontend\src\layouts\CartLayout.astro",
    "$root\frontend\src\layouts\ProductLayout.astro",
    # Pages
    "$root\frontend\src/pages\index.astro",
    "$root\frontend\src/pages\cart\index.astro",
    "$root\frontend\src/pages\cart\modify.astro",
    "$root\frontend\src/pages\cart\optimize.astro",
    "$root\frontend\src/pages\cart\checkout.astro",
    "$root\frontend\src/pages\products\index.astro",
    "$root\frontend\src/pages\products\category\[slug].astro",
    "$root\frontend\src/pages\products\[id].astro",
    "$root\frontend\src/pages\smart\search.astro",
    "$root\frontend\src/pages\smart\recommendations.astro",
    "$root\frontend\src/pages\api\ai\build-cart.js",
    "$root\frontend\src/pages\api\ai\parse-list.js",
    "$root\frontend\src/pages\api\ai\match-products.js",
    "$root\frontend\src/pages\api\ai\suggest-quantities.js",
    "$root\frontend\src/pages\api\ai\optimize-cart.js",
    "$root\frontend\src/pages\api\ai\get-alternatives.js",
    "$root\frontend\src/pages\api\cart\create.js",
    "$root\frontend\src/pages\api\cart\update.js",
    "$root\frontend\src/pages\api\cart\get.js",
    "$root\frontend\src/pages\api\cart\checkout.js",
    "$root\frontend\src/pages\api\products.js",
    "$root\frontend\src/pages\api\search.js",
    # Scripts & styles
    "$root\frontend\src/scripts/ai-cart-builder.js",
    "$root\frontend\src/scripts/smart-input.js",
    "$root\frontend\src/scripts/cart-manager.js",
    "$root\frontend\src/scripts/product-matcher.js",
    "$root\frontend\src/scripts/checkout-flow.js",
    "$root\frontend\src/scripts/animations.js",
    "$root\frontend\src/styles/global.css",
    "$root\frontend\src/styles/ai-components.css",
    "$root\frontend\src/styles/cart-builder.css",
    "$root\frontend\src/styles/transitions.css",
    "$root\frontend\src/utils/ai-api.js",
    "$root\frontend\src/utils/cart-utils.js",
    "$root\frontend\src/utils/product-utils.js",
    "$root\frontend\src/utils/constants.js",
    # Frontend config
    "$root\frontend/astro.config.mjs",
    "$root\frontend/tailwind.config.mjs",
    "$root\frontend/package.json",
    "$root\frontend/tsconfig.json"
)

New-Files $files

Write-Host "✅ List2Cart project structure created successfully!"
