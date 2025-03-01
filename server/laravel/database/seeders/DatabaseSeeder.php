<?php

namespace Database\Seeders;

// use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use App\Models\RoutePhoto;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        RouteDifficultySeeder::run();
        RouteSeeder::run();
        RoutePhotoSeeder::run();
        RouteCategorySeeder::run();
        RouteCategoryToRouteSeeder::run();
    }
}
