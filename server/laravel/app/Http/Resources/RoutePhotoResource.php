<?php

namespace App\Http\Resources;

use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Storage;

class RoutePhotoResource extends JsonResource
{
    private const PATH_IMAGE_FOLDER = '/images/';

    /**
     * Transform the resource into an array.
     *
     * @return array<string, mixed>
     */
    public function toArray(Request $request): array
    {
        return [
                $request->getSchemeAndHttpHost()
                . self::PATH_IMAGE_FOLDER
                . $this->photo_path
        ];
    }
}
